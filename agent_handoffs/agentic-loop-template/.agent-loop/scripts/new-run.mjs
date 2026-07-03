import fs from "node:fs";
import path from "node:path";

const runsDir = ".agent-runs";
fs.mkdirSync(runsDir, { recursive: true });

const now = new Date();
const runId = now.toISOString().replace(/[:.]/g, "-");
const runDir = path.join(runsDir, runId);

fs.mkdirSync(runDir, { recursive: true });

for (let i = 1; i <= 5; i += 1) {
  const iter = String(i).padStart(2, "0");
  fs.mkdirSync(path.join(runDir, `iteration-${iter}`), { recursive: true });
}

fs.writeFileSync(path.join(runsDir, "latest"), runId + "\n");
fs.writeFileSync(path.join(runDir, "run-state.json"), JSON.stringify({
  run_id: runId,
  created_at: now.toISOString(),
  iterations_expected: 5,
  status: "created"
}, null, 2) + "\n");

console.log(runId);
