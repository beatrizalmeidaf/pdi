"""Progresso e estimativa de término da rodada de publicação (scripts/run_publicacao.sh)."""
import re, sys, time
from pathlib import Path

L = Path("results/pub_logs")
N_COND = {"oracle": 27, "main": 106}
# estimativas para as fases depois da A (min), atualizadas pelos tempos reais quando existirem
EST_B, EST_C, EST_D = 120, 40, 2


def job_progress(log: Path):
    """(fração concluída, minutos decorridos) a partir das linhas 'N/M imagens (X min)'."""
    stage, frac, mins = None, {"oracle": 0.0, "main": 0.0}, 0.0
    for line in log.read_text(errors="ignore").splitlines():
        m = re.match(r"\[(oracle|main)\] device=", line)
        if m:
            stage = m.group(1)
        m = re.search(r"\] (\d+)/(\d+) imagens\s+\(([\d.]+) min\)", line)
        if m and stage:
            frac[stage] = int(m.group(1)) / int(m.group(2))
            mins = float(m.group(3))
        if re.match(r"\[(oracle|main)\] .* (salvo em|já existe)", line) and stage:
            frac[stage] = 1.0
    w = sum(N_COND[k] * frac[k] for k in frac) / sum(N_COND.values())
    return w, mins


def kind(name):
    return name.split("_")[0] if not name.startswith("adapt") else name.rsplit("_", 1)[0]


def main():
    if not (L / "jobs_a.txt").exists():
        sys.exit("rodada não iniciada (results/pub_logs/jobs_a.txt não existe)")
    phases = [l.split() for l in (L / "phases.txt").read_text().splitlines()]
    cur, t_cur = phases[-1][0], int(phases[-1][1])
    now = time.time()
    t0 = int(phases[0][1])
    print(f"fase atual: {cur}   (início da rodada há {(now - t0) / 3600:.1f} h)")
    if cur == "FIM":
        print("rodada concluída: results/pub_agregado/summary.md"); return
    fails = [p.stem for p in (L / "jobs").glob("*.log") if "FALHOU" in p.read_text(errors="ignore")]
    if fails:
        print(f"!! tarefas com falha: {fails}  (ver results/pub_logs/jobs/<nome>.log)")

    rem_a = 0.0
    if cur == "A":
        names = [l.split("|")[0] for l in (L / "jobs_a.txt").read_text().splitlines()]
        done = {n for n in names if (L / "jobs" / f"{n}.done").exists()}
        dur, running, pending = {}, {}, []
        for n in names:
            log = L / "jobs" / f"{n}.log"
            if n in done:
                txt = log.read_text(errors="ignore")
                ts = re.findall(r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\]", txt)
                if len(ts) >= 2:
                    a, b = (time.mktime(time.strptime(t, "%Y-%m-%d %H:%M:%S")) for t in (ts[0], ts[-1]))
                    dur.setdefault(kind(n), []).append((b - a) / 60)
            elif log.exists():
                f, m = job_progress(log)
                running[n] = (f, m)
                if f > 0.05:
                    dur.setdefault(kind(n), []).append(m / f)
            else:
                pending.append(n)
        avg = {k: sum(v) / len(v) for k, v in dur.items()}
        default = sum(avg.values()) / len(avg) if avg else 60.0
        slot_min = sum(max(avg.get(kind(n), default) * (1 - f), 0) for n, (f, m) in running.items())
        slot_min += sum(avg.get(kind(n), default) for n in pending)
        slots = max(min(44, len(running) + len(pending)), 1)
        rem_a = slot_min / slots
        print(f"fase A: {len(done)}/{len(names)} tarefas prontas, {len(running)} rodando, {len(pending)} na fila")
        for k, v in sorted(avg.items()):
            print(f"   duração estimada por tarefa {k:<18} {v:6.0f} min")
        if running:
            fr = sorted(running.items(), key=lambda kv: kv[1][0])
            print(f"   tarefa mais atrasada: {fr[0][0]} ({100 * fr[0][1][0]:.0f} %)")
    rem = rem_a + {"A": EST_B + EST_C + EST_D, "B": EST_B + EST_C + EST_D - (now - t_cur) / 60,
                   "C": EST_C + EST_D - (now - t_cur) / 60, "D": EST_D}[cur]
    rem = max(rem, 5)
    eta = time.strftime("%d/%m %H:%M", time.localtime(now + rem * 60))
    print(f"estimativa de término: {eta}  (faltam ~{rem / 60:.1f} h; fases B–D estimadas em ~{(EST_B + EST_C) / 60:.1f} h)")


if __name__ == "__main__":
    main()
