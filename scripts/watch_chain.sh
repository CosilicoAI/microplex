#!/bin/zsh
# Chain trigger: exits (re-invoking the agent) on the FIRST of —
#   * a new "=== CHAIN step" boundary (sanity-check the finished step)
#   * a new Traceback anywhere in the log (incl. fail-soft telemetry ones)
#   * a gate failure line ("passed=False")
#   * log silence > 12 min (hang detection)
#   * chain exit
# The agent inspects, then re-arms this watcher. One watcher, every signal.
LOG=/tmp/populace_chain.log
steps0=$(grep -c "=== CHAIN" $LOG 2>/dev/null || true)
tb0=$(grep -c "Traceback" $LOG 2>/dev/null || true)
fail0=$(grep -c "passed=False" $LOG 2>/dev/null || true)

steps0=${steps0:-0}; tb0=${tb0:-0}; fail0=${fail0:-0}
while true; do
  sleep 20
  if ! pgrep -f "[r]un_chain.sh" >/dev/null; then
    echo "TRIGGER: chain exited"
    break
  fi
  steps=$(grep -c "=== CHAIN" $LOG 2>/dev/null || true); steps=${steps:-0}
  tb=$(grep -c "Traceback" $LOG 2>/dev/null || true); tb=${tb:-0}
  fail=$(grep -c "passed=False" $LOG 2>/dev/null || true); fail=${fail:-0}
  if (( tb > tb0 )); then
    echo "TRIGGER: new Traceback (count $tb0 -> $tb)"
    break
  fi
  if (( fail > fail0 )); then
    echo "TRIGGER: gate failure logged"
    break
  fi
  if (( steps > steps0 )); then
    echo "TRIGGER: step boundary ($steps0 -> $steps)"
    break
  fi
  if [[ -f $LOG ]]; then
    age=$(( $(date +%s) - $(stat -f %m $LOG) ))
    if (( age > 720 )); then
      echo "TRIGGER: log silent for ${age}s (possible hang)"
      break
    fi
  fi
done
echo "--- state at trigger ---"
grep "=== CHAIN" $LOG | tail -2
tail -4 $LOG
