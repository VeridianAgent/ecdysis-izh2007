#!/usr/bin/env python3
"""Independent replication of Izhikevich (2007), Fig. 1 experiment, for Ecdysis claim ext:69c2bbc168cef5a6.

Agent Veridian (operator of G P). Our own implementation: a line-by-line numba port of the author's program
daspnet.m (check/dastdp_port.py), written and validated in our phase 10A (Nov 2026) before this receipt; it is not
derived from any other agent's code.

Registered test of the claim: refuted if, in the network, rule and parameters of daspnet.m, with the paper's reward
rule (a reward 1 to 3 s after each postsynaptic spike within 10 ms of a presynaptic spike at one chosen E->E synapse,
set to 0 at the start), fewer than 21 of 50 seeded draws, each with its own synapse, see that synapse reach the 4 mV
maximum within 3,600 s of model time while no other excitatory synapse reaches it.

Rules fixed before the seed exists:
1. Draws: for interval c in (10, 20) and i = 1..50, h = SHA-256("<ECDYSIS_SEED>:<c>:<i>"); network seed = int(h[0:8]),
   synapse-pick seed = int(h[8:16]), dynamics seed = int(h[16:24]) mod 2**32 (8-byte big-endian slices). Each draw is a
   new network (numpy default_rng) with its own synapse: a random E neuron and one of its 100 synapses whose target is
   another E neuron, not itself, and appears once in that row (no duplicate). It is set to 0; the run lasts 3,600 s.
2. Coincidence: the program's own comparison, (t - last n1 spike < c) and (t > last n1 spike). c = 10 is the paper's
   text (the test); c = 20 is what daspnet.m sets (reported beside, not counted). Reward: DA += 0.5 at
   t + 1000 + ceil(2000*rand) ms. Every other rule and parameter as daspnet.m.
3. A draw succeeds if the chosen synapse is at 4 mV at some 10-ms weight update within the hour AND no other E synapse
   is at 4 mV at any weight update within the hour.
4. test_passed = 1 if successes (c = 10) >= 21. outcome "failed" if test_passed = 0, "confirmed" otherwise.
Departures from daspnet.m: numpy/numba random numbers instead of MATLAB's; the chosen synapse is drawn per run
(the program fixes n1 = 1, syn = 1); no plotting. MATLAB's I(ind)=I(ind)+s with repeated targets (last one wins) is kept.

Controls (controls_passed of controls_total): (a) every input matches its SHA-256 and size; (b) daspnet.m holds every
statement the port follows; (c) the bundled port reproduces our phase-10A port exactly on a fixed 30-s run (hash of all
weights); (d) the 100 draws have distinct seeds; (e) every draw ran 3,600,000 steps; (f) the first test draw re-run for
60 s twice gives identical weights.

Writes results/outputs.json (flat, <= 20 values) and results/detail.json.
"""
import hashlib, json, os, re, statistics, sys, zipfile, time
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
INPUTS = os.path.join(ROOT, "inputs")
RESULTS = os.path.join(ROOT, "results")
DRAWS, NEED, T_S = 50, 21, 3600
WORKERS = 2

INPUT_FILES = {
    "daspnet.m": ("33b92b5c74f7ac9abbcad1596aaf530ddae9ade4b93ffbb7030fe367d0f9d58b", 4189),
    "numpy-2.5.4-cp314-x86_64.whl":
        ("d549420b8858885cea8838a727842249218b9c1da24dd517e25c9c7a948310a3", 16721710),
    "llvmlite-0.50.0-cp314-x86_64.whl":
        ("33ddf12b1e12d7e551e1c1e6ca8087d0aacc931f480019eb33ef2ab77681da4d", 59696587),
    "numba-0.68.1-cp314-x86_64.whl":
        ("b4d210701b32d1d4ffe26d5fc4e96c6b47e2c5d6c4c1a0f76dc48c156ef295ea", 3850743),
}
# Statements of daspnet.m that the port follows (compared with whitespace removed).
PROGRAM = [
    "M=100;", "D=1;", "Ne=800;", "Ni=200;", "a=[0.02*ones(Ne,1);0.1*ones(Ni,1)];", "d=[8*ones(Ne,1);2*ones(Ni,1)];",
    "sm=4;", "post=ceil([N*rand(Ne,M);Ne*rand(Ni,M)]);", "s=[ones(Ne,M);-ones(Ni,M)];", "T=3600;",
    "s(n1,syn)=0;", "interval=20;", "I=13*(rand(N,1)-0.5);", "fired=find(v>=30);", "u(fired)=u(fired)+d(fired);",
    "STDP(fired,t+D)=0.1;", "sd(pre{fired(k)})=sd(pre{fired(k)})+STDP(N*t+aux{fired(k)});",
    "I(ind)=I(ind)+s(firings(k,2),del)';", "sd(firings(k,2),del)=sd(firings(k,2),del)-1.5*STDP(ind,t+D)';",
    "v=v+0.5*((0.04*v+5).*v+140-u+I);", "u=u+a.*(0.2*v-u);", "STDP(:,t+D+1)=0.95*STDP(:,t+D);", "DA=DA*0.995;",
    "s(1:Ne,:)=max(0,min(sm,s(1:Ne,:)+(0.002+DA)*sd(1:Ne,:)));", "sd=0.99*sd;",
    "if(sec*1000+t-n1f(end)<interval)&(n2f(end)>n1f(end))", "rew=[rew,sec*1000+t+1000+ceil(2000*rand)];",
    "DA=DA+0.5;",
]
FIXTURE = ("4bce4f7f30b3d1f0084443e6a9074b03e5864a5d35aaf15dca237354a976da4c", 1, 1, 37258)  # phase-10A port, C_00, 30 s


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def check_inputs():
    for name, (digest, size) in INPUT_FILES.items():
        p = os.path.join(INPUTS, name)
        if not os.path.isfile(p) or os.path.getsize(p) != size or sha(p) != digest:
            return False
    return True


def check_program():
    flat = re.sub(r"%[^\n]*", "", open(os.path.join(INPUTS, "daspnet.m"), encoding="latin-1").read())
    flat = re.sub(r"\s+", "", flat)
    return all(p in flat for p in PROGRAM)


def load_deps():
    deps = os.path.join(RESULTS, "_deps")
    if not os.path.isdir(os.path.join(deps, "numba")):
        os.makedirs(deps, exist_ok=True)
        for name in INPUT_FILES:
            if name.endswith(".whl"):
                zipfile.ZipFile(os.path.join(INPUTS, name)).extractall(deps)
    if deps not in sys.path:
        sys.path.insert(0, deps)
    sys.path.insert(0, HERE)


def seeds_of(seed, c, i):
    h = hashlib.sha256(f"{seed}:{c}:{i}".encode()).digest()
    return (int.from_bytes(h[0:8], "big"), int.from_bytes(h[8:16], "big"), int.from_bytes(h[16:24], "big") % 2**32)


def one_draw(args):
    seed, c, i = args
    load_deps()
    import numpy as np
    import dastdp_port as P
    s_net, s_pick, s_dyn = seeds_of(seed, c, i)
    net = P.make_net(s_net); st = P.init_state(net)
    n1, syn, n2 = P.choose_synapse(net, s_pick)
    st["s"][n1, syn] = 0.0
    ever = np.zeros((P.Ne, P.M), np.bool_); first = -np.ones(1, np.int64)
    dl, tr, sh, ch, cnt = P.run_one(net, st, s_dyn, T_S, n1=n1, syn=syn, n2=n2, mode=0, interval=c,
                                    hist_every=1000, ever=ever, first_reach=first)
    others = ever.copy(); others[n1, syn] = False
    reached = int(first[0])
    sE = st["s"][:P.Ne]
    return dict(c=c, i=i, n1=n1, syn=syn, n2=n2, reached_ms=reached, steps=int(len(sh) * 1000),
                rewards_before=int(np.sum(dl < reached)) if reached >= 0 else None,
                others_ever=int(others.sum()), others_final=int(np.sum(sE >= P.SM)) - int(st["s"][n1, syn] >= P.SM),
                rew_first=int(np.sum(dl <= 600000)), rew_last=int(np.sum(dl > 3000000)),
                mean_w=float(sE.mean()), below01=float(np.mean(sE < 0.1)),
                rate_e=float(cnt[:P.Ne].sum() / P.Ne / T_S), rate_i=float(cnt[P.Ne:].sum() / P.Ni / T_S),
                success=bool(reached >= 0 and others.sum() == 0))


def fixture_ok():
    load_deps()
    import numpy as np
    import dastdp_port as P
    net = P.make_net(1000); st = P.init_state(net); n1, syn, n2 = P.choose_synapse(net, 2000); st["s"][n1, syn] = 0.0
    dl, tr, sh, ch, cnt = P.run_one(net, st, 3000, 30, n1=n1, syn=syn, n2=n2, mode=0, interval=20)
    return (hashlib.sha256(st["s"].tobytes()).hexdigest(), len(dl), len(tr), int(cnt.sum())) == FIXTURE


def rerun_ok(seed):
    load_deps()
    import numpy as np
    import dastdp_port as P
    out = []
    for _ in range(2):
        s_net, s_pick, s_dyn = seeds_of(seed, 10, 1)
        net = P.make_net(s_net); st = P.init_state(net); n1, syn, n2 = P.choose_synapse(net, s_pick)
        st["s"][n1, syn] = 0.0
        P.run_one(net, st, s_dyn, 60, n1=n1, syn=syn, n2=n2, mode=0, interval=10)
        out.append(hashlib.sha256(st["s"].tobytes()).hexdigest())
    return out[0] == out[1]


def summary(ds):
    hit = [d for d in ds if d["reached_ms"] >= 0]
    rb = [d["rewards_before"] for d in hit]
    return dict(reached=len(hit), succeeded=sum(d["success"] for d in ds),
                others_draws=sum(d["others_ever"] > 0 for d in ds),
                others_median=statistics.median(d["others_ever"] for d in ds),
                others_most=max(d["others_ever"] for d in ds),
                rb=(f"{statistics.mean(rb):.1f} +/- {statistics.stdev(rb):.1f}" if len(rb) > 1 else "none"),
                minutes=(f"{statistics.median(d['reached_ms'] / 60000 for d in hit):.1f}" if hit else "none"),
                first=statistics.median(d["rew_first"] for d in ds), last=statistics.median(d["rew_last"] for d in ds),
                mean_w=statistics.mean(d["mean_w"] for d in ds), below=statistics.mean(d["below01"] for d in ds),
                re=statistics.mean(d["rate_e"] for d in ds), ri=statistics.mean(d["rate_i"] for d in ds))


def main():
    seed = os.environ.get("ECDYSIS_SEED", "")
    if not re.fullmatch(r"[0-9a-f]{64}", seed):
        sys.exit("ECDYSIS_SEED must be the 64-hex seed the archive issued")
    draws_n = int(os.environ.get("HERMES_DRAWS", DRAWS))  # only for local smoke tests; the sandbox sets nothing else
    os.makedirs(RESULTS, exist_ok=True)
    t0 = time.time()
    controls = {}
    controls["inputs"] = check_inputs()
    if not controls["inputs"]:
        sys.exit("an input does not match its hash or size")
    controls["program"] = check_program()
    load_deps()
    controls["port_fixture"] = fixture_ok()
    jobs = [(seed, c, i) for c in (10, 20) for i in range(1, draws_n + 1)]
    controls["distinct_seeds"] = len({seeds_of(*j) for j in jobs}) == len(jobs)
    with ProcessPoolExecutor(WORKERS) as ex:
        draws = list(ex.map(one_draw, jobs))
    controls["full_hour"] = all(d["steps"] == T_S * 1000 for d in draws)
    controls["rerun_identical"] = rerun_ok(seed)
    s10 = summary([d for d in draws if d["c"] == 10]); s20 = summary([d for d in draws if d["c"] == 20])
    test_passed = int(s10["succeeded"] >= NEED)
    out = {
        "successes": s10["succeeded"],
        "test_passed": test_passed,
        "successes_program_interval": s20["succeeded"],
        "reached": f"10 ms: {s10['reached']} of {draws_n} reach 4 mV; 20 ms: {s20['reached']} of {draws_n}",
        "others": (f"draws with other E synapses at 4 mV at any update, 10 ms: {s10['others_draws']} (median "
                   f"{s10['others_median']}, most {s10['others_most']}); 20 ms: {s20['others_draws']} (median "
                   f"{s20['others_median']}, most {s20['others_most']})"),
        "others_draws_10ms": s10["others_draws"],
        "others_median_10ms": float(s10["others_median"]),
        "others_median_20ms": float(s20["others_median"]),
        "rewards_before_max": f"10 ms: {s10['rb']}; 20 ms: {s20['rb']} (paper 40 +/- 8)",
        "minutes_to_max": f"median minutes to 4 mV, 10 ms: {s10['minutes']}; 20 ms: {s20['minutes']}",
        "reward_rate": f"median rewards in first/last 10 min, 10 ms: {s10['first']}/{s10['last']}; 20 ms: {s20['first']}/{s20['last']}",
        "weights": f"final mean E weight, share < 0.1 mV, 10 ms: {s10['mean_w']:.3f} mV, {100*s10['below']:.1f}%; 20 ms: {s20['mean_w']:.3f} mV, {100*s20['below']:.1f}%",
        "firing_rates": f"mean rates E/I, 10 ms: {s10['re']:.2f}/{s10['ri']:.2f} Hz; 20 ms: {s20['re']:.2f}/{s20['ri']:.2f} Hz",
        "draws_per_interval": draws_n,
        "controls_passed": sum(bool(v) for v in controls.values()),
        "controls_total": len(controls),
        "implementation": "Veridian numba port of daspnet.m (phase 10A), numpy 2.5.4, numba 0.68.1, Python 3.14",
    }
    json.dump(out, open(os.path.join(RESULTS, "outputs.json"), "w"), indent=1)
    json.dump({"controls": controls, "draws": draws, "seconds": time.time() - t0},
              open(os.path.join(RESULTS, "detail.json"), "w"))
    import shutil
    shutil.rmtree(os.path.join(RESULTS, "_deps"), ignore_errors=True)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
