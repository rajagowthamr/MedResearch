#!/usr/bin/env bash
#
# Edit on the Mac, train on the RTX 3060, in one command.
#
# WHY NOT git clone
#   A clone is a frozen snapshot: every code tweak would need commit + push +
#   pull before the GPU sees it. rsync sends only what CHANGED, so after the
#   first push a model.py edit reaches the box in about a second. It also picks
#   up files git does not track (requirements.txt, this script).
#
# WHY init AND train ARE SEPARATE
#   `init` pushes the corpus, mlflow.db and mlruns/ ONE TIME. `train` deliberately
#   does NOT — because after the first run those directories are produced BY the
#   server, and re-pushing the Mac's copies would overwrite the checkpoints and
#   run history the GPU just generated. Sync the inputs once; pull the outputs back.
#
#   ./gpu.sh init      first-time setup: files, venv, torch, mlflow path fix
#   ./gpu.sh smoke     sync code, 200-step test run in the foreground (~2 min)
#   ./gpu.sh train     sync code, launch the real run detached (~2 h)
#   ./gpu.sh progress  loss curve + steps/sec + ETA   <- the one to use
#   ./gpu.sh ui        MLflow UI, tunnelled to http://localhost:5555
#   ./gpu.sh ui-stop   shut the MLflow server down on the box
#   ./gpu.sh watch     live GPU temperature, utilisation and memory
#   ./gpu.sh log       follow train_v5.log (buffered unless launched with -u)
#   ./gpu.sh pull      bring checkpoints + mlflow history back to the Mac
#   ./gpu.sh shell     plain ssh into the box
set -euo pipefail
cd "$(dirname "$0")"

HOST=ekko@gpu.aitranscribeapp.com
REMOTE=MedResearch                      # relative to the remote ~
MAC_ROOT=/Users/rajagowtham/Documents/MedResearch   # the path baked into mlflow.db

# The architecture. ~30M params vs the Mac model's 4.86M.
#
# Sized for 12GB, not the 16GB T4 that train_gpu.ipynb assumes:
#   activations  ~5.4 GB   8 blocks x ~670MB at B=64, T=512, C=512
#   optim state  ~0.5 GB   30M params x 16B (fp32 param + grad + 2 Adam moments)
#   corpus       ~0.1 GB   60.7M tokens as int16, parked on the GPU
#   context      ~0.5 GB
#   peak         ~6.8 GB of 12 GB
# On CUDA OOM, halve BATCH_SIZE. Do NOT cut BLOCK_SIZE — the context window is
# what makes the model better; batch size only changes gradient noise.
#
# 20000 x 64 x 512 = 655M tokens against 54.6M train tokens (~12 passes).
# Chinchilla-optimal for 30M params is ~600M, so this lands on the compute-optimal
# point. The Mac run saw 41M tokens — less than one pass.
ARCH=(
  BLOCK_SIZE=512 N_EMBD=512 N_HEAD=8 N_LAYER=8
  BATCH_SIZE=64 DROPOUT=0.15
  LEARNING_RATE=6e-4 MIN_LR=6e-5 WARMUP_ITERS=500
)

# Be explicit rather than trusting MLflow to auto-detect ./mlflow.db. If that
# detection misses, MLflow silently starts a FRESH ./mlruns file store: the run
# looks fine, the registry step fails, and the new run never appears beside the
# old ones. Naming the URI removes the guess.
ENV_PREFIX="cd ~/$REMOTE && source venv/bin/activate && export MLFLOW_TRACKING_URI=sqlite:///mlflow.db"

# The project is a src/-layout package now, so `python scripts/train_gpt.py`
# cannot find `medresearch` unless it has been installed. `pip install -e .` is
# idempotent and takes about a second once satisfied, so it runs before every
# launch rather than only in init -- otherwise adding a module to the package
# would fail on the box with ModuleNotFoundError long after setup "succeeded".
INSTALL="venv/bin/pip install -qe ."
TRAIN_ENTRY="scripts/train_gpt.py"

# NOTE ON FLAGS: macOS ships openrsync ("rsync 2.6.9 compatible"), which predates
# --info=progress2/--info=stats1 and dies on them. Everything here sticks to -v
# and --progress, which both openrsync and rsync 3.x understand.

# Code only. Everything the GPU produces is excluded so a sync never clobbers it.
sync_code() {
  echo "--> syncing code to $HOST:~/$REMOTE"
  rsync -avz \
    --exclude venv/ --exclude .git/ --exclude __pycache__/ --exclude '*.pyc' \
    --exclude mlruns/ --exclude checkpoints/ --exclude mlflow.db \
    --exclude '*.log' \
    ./ "$HOST:$REMOTE/"
}

case "${1:-}" in

init)
  echo "==> 1/4 pushing code + corpus + existing mlflow history (~320MB, once)"
  # mlruns/ and mlflow.db ARE included here, and only here: this is what carries
  # your 16 previous runs onto the box so new runs land in the same history.
  rsync -az --progress \
    --exclude venv/ --exclude .git/ --exclude __pycache__/ --exclude '*.pyc' \
    ./ "$HOST:$REMOTE/"

  echo "==> 2/4 building venv and installing torch"
  # Only these three are needed to train. On Linux the torch wheel BUNDLES the
  # CUDA runtime, so no CUDA toolkit and no --index-url — the NVIDIA driver
  # (595.84, already installed) is the only system requirement.
  # NO SUDO, deliberately. This box (Ubuntu 26.04, Python 3.14) does not have the
  # python3-venv package, so `python3 -m venv venv` builds a venv with NO pip in
  # it — and it exits 0 while doing so, which is the nasty part: the failure only
  # surfaces later as "bash: venv/bin/pip: No such file or directory".
  # `--without-pip` makes that explicit, then get-pip.py installs pip INTO the
  # venv from upstream. Same end state as `apt install python3-venv`, minus the
  # root password, which matters because this script has to run unattended.
  ssh "$HOST" "set -e
    cd ~/$REMOTE
    rm -rf venv
    python3 -m venv --without-pip venv
    curl -fsSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
    venv/bin/python /tmp/get-pip.py -q
    venv/bin/pip --version
    # Only these three are needed to train. On Linux the torch wheel BUNDLES the
    # CUDA runtime, so no CUDA toolkit and no --index-url — the NVIDIA driver
    # (595.84, already installed) is the only system requirement.
    venv/bin/pip install torch==2.13.0 mlflow==3.14.0 numpy==2.5.1
    venv/bin/pip install -qe ."

  echo "==> 3/4 repointing mlflow history from the Mac path to this box"
  # Every one of the 16 existing runs stores its artifact folder as an absolute
  # Mac path, which does not exist on Ubuntu. Left alone, log_artifact() fails
  # and the new run never registers. sqlite3 ships inside python; no apt needed.
  ssh "$HOST" "$ENV_PREFIX && python - <<'PY'
import sqlite3, os
OLD, NEW = '$MAC_ROOT', os.path.abspath('.')
db = sqlite3.connect('mlflow.db')
db.execute('UPDATE experiments SET artifact_location=replace(artifact_location,?,?)', (OLD, NEW))
db.execute('UPDATE runs SET artifact_uri=replace(artifact_uri,?,?)', (OLD, NEW))
db.commit()
print(db.execute('SELECT count(*) FROM runs WHERE artifact_uri LIKE ?', (NEW+'%',)).fetchone()[0], 'runs repointed to', NEW)
PY"

  echo "==> 4/4 verifying CUDA"
  # A CPU fallback is IDENTICAL apart from being ~40x slower. Nothing errors.
  # So fail loudly here rather than discover it three hours in.
  ssh "$HOST" "$ENV_PREFIX && python - <<'PY'
import torch
print('torch      :', torch.__version__)
print('cuda build :', torch.version.cuda)
print('gpu        :', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')
print('bf16       :', torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)
assert torch.cuda.is_available(), 'No CUDA. Check nvidia-smi, and that pip installed the Linux wheel and not the CPU one.'
print('OK - ready to train')
PY"
  echo
  echo "Ready.  ./gpu.sh smoke   (2 min sanity run)   then   ./gpu.sh train"
  ;;

smoke)
  # 200 steps in the foreground. Catches a broken setup in 2 minutes instead of
  # 3 hours. Look for: Device: cuda / ~30M params / bfloat16 / "Registered".
  sync_code
  ssh -t "$HOST" "$ENV_PREFIX && $INSTALL && VERSION=smoke-3060 MAX_ITERS=200 EVAL_INTERVAL=100 ${ARCH[*]} python -u $TRAIN_ENTRY"
  ;;

train)
  sync_code
  # nohup + & so closing this laptop, or the ssh session dropping, does not kill
  # a 3-hour run. train.py writes the best checkpoint at every eval_interval, so
  # even an abrupt death leaves you a usable model.
  ssh "$HOST" "$ENV_PREFIX && $INSTALL && VERSION=v5-large MAX_ITERS=20000 EVAL_INTERVAL=500 ${ARCH[*]} \
    nohup python -u $TRAIN_ENTRY > train_v5.log 2>&1 & echo started pid \$!"
  echo "launched. follow it with:  ./gpu.sh log"
  ;;

log)   ssh -t "$HOST" "tail -f ~/$REMOTE/train_v5.log" ;;

progress)
  # Reads the loss curve out of MLflow rather than the log. This is the reliable
  # one: python block-buffers stdout when it is redirected to a file, so a run
  # launched without `python -u` writes nothing to train_v5.log for minutes at a
  # time and `log` looks frozen while training is perfectly fine. MLflow gets
  # every metric at eval time regardless of buffering.
  ssh "$HOST" "$ENV_PREFIX && python - <<'PY'
import sqlite3
db = sqlite3.connect('mlflow.db')
run = db.execute(\"select run_uuid,name from runs where name like 'v5-large%' order by start_time desc limit 1\").fetchone()
if not run: raise SystemExit('no v5-large run found')
print('run:', run[1])
d = {}
for s, k, v in db.execute(
        \"select step,key,value from metrics where run_uuid=? and key in ('train_loss','val_loss') order by step\", (run[0],)):
    d.setdefault(s, {})[k] = v
for s in sorted(d):
    print(f\"step {s:>6}: train {d[s].get('train_loss',0):.4f} | val {d[s].get('val_loss',0):.4f}\")
best = min((r['val_loss'] for r in d.values() if 'val_loss' in r), default=None)
if best: print(f\"\nbest val so far {best:.4f}   (Mac v2 best was 1.2296)   {max(d)}/20000 steps\")

# Rate straight off MLflow's per-metric timestamps (unix ms). train.py does print
# its own elapsed time at each eval, but stdout is block-buffered into the log
# file, so that number is invisible until the run ends. These rows are not.
t = db.execute(\"select step,timestamp from metrics where run_uuid=? and key='val_loss' order by step\", (run[0],)).fetchall()
if len(t) > 1:
    rate = (t[-1][0] - t[0][0]) / ((t[-1][1] - t[0][1]) / 1000)
    left = (20000 - t[-1][0]) / rate
    print(f\"rate {rate:.2f} steps/sec | ~{left/60:.0f} min remaining\" if left > 0 else 'FINISHED')
PY"
  ;;

watch) ssh -t "$HOST" "nvidia-smi -l 2" ;;

pull)
  # The reverse trip: outputs come back, and the paths get rewritten to the Mac
  # so `mlflow ui` here can find the artifacts again.
  echo "--> pulling checkpoints + mlflow history"

  # mlflow.db comes FIRST and over scp, not rsync, and that ordering is the fix
  # for a real failure: openrsync died mid-transfer on the ~300MB mlruns/ tree
  # ("unexpected end of file" through the cloudflared tunnel) and took the rest
  # of the pull down with it, so the database -- the one thing that actually
  # holds the metrics and the registry -- never arrived. It is a couple of MB;
  # copy it before anything large can fail.
  scp -q "$HOST:$REMOTE/mlflow.db" ./mlflow.db && echo "    mlflow.db OK"

  rsync -avz "$HOST:$REMOTE/checkpoints/" ./checkpoints/

  # mlruns/ is only artifact COPIES of the checkpoints above, so a failure here
  # costs you nothing that matters. --partial keeps what transferred so a retry
  # resumes, and `|| true` stops it aborting the path rewrite below.
  rsync -avz --partial "$HOST:$REMOTE/mlruns/" ./mlruns/ \
    || echo "    WARNING: mlruns/ incomplete (artifact copies only; metrics are safe in mlflow.db). Re-run ./gpu.sh pull to resume."

  python3 - <<PY
import sqlite3, os
OLD, NEW = os.path.expanduser('/home/ekko/$REMOTE'), '$MAC_ROOT'
db = sqlite3.connect('mlflow.db')
db.execute('UPDATE experiments SET artifact_location=replace(artifact_location,?,?)', (OLD, NEW))
db.execute('UPDATE runs SET artifact_uri=replace(artifact_uri,?,?)', (OLD, NEW))
db.commit()
print(db.execute('SELECT count(*) FROM runs WHERE artifact_uri LIKE ?', (NEW+'%',)).fetchone()[0], 'runs repointed back to the Mac')
PY
  echo "done.  mlflow ui --backend-store-uri sqlite:///mlflow.db"
  ;;

ui)
  # MLflow UI for the run happening RIGHT NOW, tunnelled from the box.
  #
  # Local port is 5555, not 5000: macOS binds 5000 to the AirPlay Receiver by
  # default, so -L 5000:... appears to work and then serves you AirPlay.
  # Remote stays 5000, and --host 127.0.0.1 keeps it off the public interface —
  # the tunnel is the only way in.
  # Start the server only if one is not already up, then hold the tunnel open.
  # Blindly launching a second `mlflow ui` just fails to bind 5000 and spews a
  # traceback, while the tunnel quietly forwards to the FIRST one anyway — so it
  # half-works and looks broken. Check first instead.
  ssh "$HOST" "pgrep -f '[u]vicorn.*mlflow' >/dev/null || {
      cd ~/$REMOTE && setsid venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db \
        --host 127.0.0.1 --port 5000 > /tmp/ui.log 2>&1 < /dev/null &
      sleep 12; }"
  echo "MLflow UI ->  http://localhost:5555     (ctrl-C closes the tunnel; training and the server keep running)"
  ssh -N -L 5555:localhost:5000 "$HOST"
  ;;

ui-stop) ssh "$HOST" "pkill -f '[u]vicorn.*mlflow'; pkill -f '[v]env/bin/mlflow'; echo 'mlflow server stopped'" ;;

shell) ssh -t "$HOST" "cd ~/$REMOTE && exec \$SHELL -l" ;;

*)
  sed -n '3,22p' "$0" | sed 's/^# \{0,1\}//'
  exit 1 ;;
esac
