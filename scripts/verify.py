"""Verificacao ponta a ponta para avaliacao do trabalho (um unico comando).

Etapas:
  1. Testes unitarios (pytest).
  2. Checagem de artefatos (video, modelo de campo, pesos do detector).
  3. Execucao do pipeline completo em poucos frames.
  4. Validacao do video e do JSON de metricas gerados.

Uso:
    python -m scripts.verify                  # completo (30 frames)
    python -m scripts.verify --frames 60
    python -m scripts.verify --skip-pipeline  # so testes + artefatos

Codigo de saida: 0 = tudo OK; 1 = alguma verificacao falhou. Relatorio em runs/verify_report.json.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys

from src.utils import PROJECT_ROOT, get_logger, resolve

log = get_logger("verify")

REQUIRED_KEYS = (
    "frames_processados",
    "fps_pipeline",
    "posse_de_bola",
    "distancia_total_equipes_m",
    "velocidade_maxima_jogo_kmh",
    "jogadores",
)


def _run(cmd: list[str]) -> int:
    log.info("$ %s", " ".join(cmd))
    return subprocess.run(cmd, cwd=PROJECT_ROOT).returncode


def validate_outputs(video: str, metrics: str, expected_frames: int) -> list[str]:
    """Retorna a lista de problemas encontrados nas saidas (vazia = OK)."""
    import cv2

    problems: list[str] = []
    vp, mp = resolve(video), resolve(metrics)

    if not vp.exists() or vp.stat().st_size == 0:
        problems.append(f"video de saida ausente/vazio: {vp}")
    else:
        cap = cv2.VideoCapture(str(vp))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if n < expected_frames:
            problems.append(f"video de saida com {n} frames (esperado >= {expected_frames})")

    if not mp.exists():
        problems.append(f"JSON de metricas ausente: {mp}")
    else:
        data = json.loads(mp.read_text(encoding="utf-8"))
        missing = [k for k in REQUIRED_KEYS if k not in data]
        if missing:
            problems.append(f"chaves ausentes no JSON: {missing}")
        elif data["frames_processados"] < expected_frames:
            problems.append(f"frames_processados={data['frames_processados']} < {expected_frames}")
        else:
            if data["velocidade_maxima_jogo_kmh"] > 40.0:
                problems.append("velocidade maxima acima do filtro de 40 km/h")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description="Verificacao ponta a ponta do projeto.")
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--video", default="data/121364_0.mp4")
    ap.add_argument("--skip-pipeline", action="store_true")
    ap.add_argument("--skip-tests", action="store_true")
    a = ap.parse_args()

    report: dict[str, object] = {}
    out_video, out_json = "runs/verify_output.mp4", "runs/verify_metrics.json"

    if not a.skip_tests:
        report["pytest"] = _run([sys.executable, "-m", "pytest", "-q"]) == 0

    report["assets"] = _run([sys.executable, "-m", "scripts.setup_assets"]) == 0

    if not a.skip_pipeline:
        if report["assets"]:
            rc = _run([
                sys.executable, "-m", "scripts.run_pipeline",
                "--video", a.video, "--max-frames", str(a.frames),
                "--output", out_video, "--metrics-output", out_json,
            ])
            report["pipeline"] = rc == 0
            if rc == 0:
                problems = validate_outputs(out_video, out_json, a.frames)
                report["outputs_ok"] = not problems
                report["problems"] = problems
                for p in problems:
                    log.error("PROBLEMA: %s", p)
        else:
            report["pipeline"] = False
            log.error("Pipeline pulado: artefatos ausentes (veja mensagens acima).")

    ok = all(v for k, v in report.items() if isinstance(v, bool))
    report["ok"] = ok
    rp = resolve("runs/verify_report.json")
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Relatorio: %s -> %s", rp, "APROVADO" if ok else "REPROVADO")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
