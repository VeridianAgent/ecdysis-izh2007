"""[Ecdysis bundle copy of exp10a_rstdp/dastdp.py: only change = `ever` and `first_reach` bookkeeping, which never feeds the dynamics.]
Port line-by-line of Izhikevich's daspnet.m (ModelDB 84167, orig/daspnet.m) to numba.

Order of operations per 1-ms step t (identical to daspnet.m):
 1. I = 13*(rand-0.5)
 2. fired = v>=30 ; v=-65 ; u+=d
 3. STDP(fired, t+D) = 0.1           -> trace column "nxt"
 4. for each fired (post) neuron: sd(pre syn) += STDP(pre, t)   -> column "cur" (LTP)
 5. for each neuron fired at t (D=1): I(post)+=s ; sd -= 1.5*STDP(post, t+D)   (LTD)
    MATLAB quirk I(ind)=I(ind)+s with duplicate targets: last occurrence wins (dupmask).
 6. two half-steps of v, one of u
 7. STDP(:,t+D+1) = 0.95*STDP(:,t+D)
 8. DA*=0.995 ; every 10 ms: s(E,:) = clip(s + (0.002+DA)*sd, 0, sm); sd*=0.99
 9. reward bookkeeping (contingent / replay / none), DA += 0.5 when a reward is due.
Time units: ms. Everything else as in daspnet.m.
"""
import numpy as np
from numba import njit

Ne, Ni = 800, 200
N = Ne + Ni
M = 100
SM = 4.0


def make_net(seed_net):
    rng = np.random.default_rng(seed_net)
    post = np.empty((N, M), np.int64)
    post[:Ne] = np.floor(N * rng.random((Ne, M))).astype(np.int64)      # ceil(N*rand) 1-based
    post[Ne:] = np.floor(Ne * rng.random((Ni, M))).astype(np.int64)
    s = np.vstack([np.ones((Ne, M)), -np.ones((Ni, M))])
    # pre lists: excitatory synapses onto each neuron (find(post==i & s>0)), column-major order as MATLAB
    pre_ptr = np.zeros(N + 1, np.int64)
    entries = [[] for _ in range(N)]
    for m in range(M):              # MATLAB linear index is column-major: loop m outer, p inner
        for p in range(Ne):
            entries[post[p, m]].append((p, m))
    pre_p, pre_m = [], []
    for i in range(N):
        pre_ptr[i + 1] = pre_ptr[i] + len(entries[i])
        for (p, m) in entries[i]:
            pre_p.append(p); pre_m.append(m)
    pre_p = np.array(pre_p, np.int64); pre_m = np.array(pre_m, np.int64)
    # dupmask[f,m] = True if m is the LAST occurrence of target post[f,m] in row f
    dupmask = np.zeros((N, M), np.bool_)
    for f in range(N):
        seen = set()
        for m in range(M - 1, -1, -1):
            if post[f, m] not in seen:
                dupmask[f, m] = True
                seen.add(post[f, m])
    a = np.r_[0.02 * np.ones(Ne), 0.1 * np.ones(Ni)]
    d = np.r_[8 * np.ones(Ne), 2 * np.ones(Ni)]
    return dict(post=post, s=s, pre_ptr=pre_ptr, pre_p=pre_p, pre_m=pre_m, dupmask=dupmask, a=a, d=d)


def choose_synapse(net, seed_pick):
    """Paper: 'randomly choose a synapse that connects 2 excitatory neurons'. Declared choice:
    pre != post and the target appears only once in the presynaptic row (no duplicate synapse)."""
    rng = np.random.default_rng(seed_pick)
    post = net['post']
    while True:
        n1 = int(rng.integers(Ne)); syn = int(rng.integers(M)); n2 = int(post[n1, syn])
        if n2 < Ne and n2 != n1 and np.sum(post[n1] == n2) == 1:
            return n1, syn, n2


def init_state(net):
    st = dict(v=-65.0 * np.ones(N), u=0.2 * (-65.0) * np.ones(N), sd=np.zeros((N, M)),
              cur=np.zeros(N), nxt=np.zeros(N), DA=np.zeros(1), s=net['s'].copy())
    return st


@njit(cache=False)
def _step_core(t, I, v, u, s, sd, cur, nxt, post, pre_ptr, pre_p, pre_m, dupmask, a, d,
               fired_buf, use_dup):
    # 2. fired
    nf = 0
    for i in range(N):
        if v[i] >= 30.0:
            fired_buf[nf] = i; nf += 1
    for k in range(nf):
        i = fired_buf[k]
        v[i] = -65.0
        u[i] += d[i]
        nxt[i] = 0.1                       # 3.
    # 4. LTP: post neuron fired -> pre traces at column t
    for k in range(nf):
        i = fired_buf[k]
        for q in range(pre_ptr[i], pre_ptr[i + 1]):
            p = pre_p[q]
            sd[p, pre_m[q]] += cur[p]
    # 5. delivery + LTD (D=1: spikes fired at this step)
    for k in range(nf):
        f = fired_buf[k]
        for m in range(M):
            tg = post[f, m]
            if (not use_dup) or dupmask[f, m]:
                I[tg] += s[f, m]
            sd[f, m] -= 1.5 * nxt[tg]
    # 6. neurons
    for i in range(N):
        vi = v[i]
        vi = vi + 0.5 * ((0.04 * vi + 5.0) * vi + 140.0 - u[i] + I[i])
        vi = vi + 0.5 * ((0.04 * vi + 5.0) * vi + 140.0 - u[i] + I[i])
        v[i] = vi
        u[i] = u[i] + a[i] * (0.2 * vi - u[i])
    # 7. trace shift
    for i in range(N):
        cur[i] = nxt[i]
        nxt[i] = 0.95 * nxt[i]
    return nf


@njit(cache=False)
def run_fig1(seed_dyn, T_ms, t0, v, u, s, sd, cur, nxt, DAarr, post, pre_ptr, pre_p, pre_m, dupmask,
             a, d, n1, syn, n2, mode, interval, dlo, dhi, replay, use_dup, base, rew_amp,
             hist_every, plastic_E_only, ever, first_reach):
    """mode 0 = contingent reward (daspnet.m), 1 = replay given reward times (yoked), 2 = no reward.
    Reward delay = dlo + ceil((dhi-dlo)*rand) ms (daspnet.m: 1000+ceil(2000*rand)).
    Returns arrays: reward delivery times, coincidence (trigger) times, s history, spike count per neuron."""
    np.random.seed(seed_dyn)
    I = np.zeros(N)
    fired_buf = np.zeros(N, np.int64)
    DA = DAarr[0]
    n1f = -100000
    pending = np.zeros(100000, np.int64); npend = 0
    delivered = np.zeros(100000, np.int64); ndel = 0
    trig = np.zeros(100000, np.int64); ntrig = 0
    nh = T_ms // hist_every
    shist = np.zeros(nh)
    chist = np.zeros(nh)
    counts = np.zeros(N, np.int64)
    ri = 0
    for tt in range(T_ms):
        t = t0 + tt + 1
        for i in range(N):
            I[i] = 13.0 * (np.random.random() - 0.5)
        nf = _step_core(t, I, v, u, s, sd, cur, nxt, post, pre_ptr, pre_p, pre_m, dupmask, a, d,
                        fired_buf, use_dup)
        DA = DA * 0.995
        if t % 10 == 0:
            for p in range(Ne):
                for m in range(M):
                    x = s[p, m] + (base + DA) * sd[p, m]
                    if x < 0.0:
                        x = 0.0
                    elif x > SM:
                        x = SM
                    s[p, m] = x
                    if x >= SM:
                        ever[p, m] = True            # [Ecdysis bundle] any E synapse at 4 mV at any update
                        if p == n1 and m == syn and first_reach[0] < 0:
                            first_reach[0] = t       # [Ecdysis bundle] exact first update at 4 mV
            for p in range(N):
                for m in range(M):
                    sd[p, m] *= 0.99
        f1 = False; f2 = False
        for k in range(nf):
            counts[fired_buf[k]] += 1
            if fired_buf[k] == n1:
                f1 = True
            if fired_buf[k] == n2:
                f2 = True
        if f1:
            n1f = t
        if mode == 0 and f2:
            # (sec*1000+t - n1f(end) < interval) & (n2f(end) > n1f(end))
            if (t - n1f < interval) and (t > n1f):
                trig[ntrig] = t; ntrig += 1
                pending[npend] = t + dlo + int(np.ceil((dhi - dlo) * np.random.random())); npend += 1
        give = False
        if mode == 0:
            for q in range(npend):
                if pending[q] == t:
                    give = True
        elif mode == 1:
            while ri < replay.shape[0] and replay[ri] <= t:
                if replay[ri] == t:
                    give = True
                ri += 1
        if give:
            DA += rew_amp
            delivered[ndel] = t; ndel += 1
        if tt % hist_every == 0:
            shist[tt // hist_every] = s[n1, syn]
            chist[tt // hist_every] = sd[n1, syn]
    DAarr[0] = DA
    return delivered[:ndel].copy(), trig[:ntrig].copy(), shist, chist, counts


def run_one(net, st, seed_dyn, T_s, t0_ms=0, n1=0, syn=0, n2=0, mode=0, interval=20, dlo=1000, dhi=3000,
            replay=None, use_dup=True, base=0.002, rew_amp=0.5, hist_every=100, ever=None, first_reach=None):
    if ever is None:
        ever = np.zeros((Ne, M), np.bool_)
    if first_reach is None:
        first_reach = -np.ones(1, np.int64)
    if replay is None:
        replay = np.zeros(0, np.int64)
    return run_fig1(seed_dyn, int(T_s * 1000), t0_ms, st['v'], st['u'], st['s'], st['sd'], st['cur'], st['nxt'],
                    st['DA'], net['post'], net['pre_ptr'], net['pre_p'], net['pre_m'], net['dupmask'],
                    net['a'], net['d'], n1, syn, n2, mode, interval, dlo, dhi, np.asarray(replay, np.int64),
                    use_dup, base, rew_amp, hist_every, True, ever, first_reach)
