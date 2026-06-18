import { execSync } from "node:child_process";

try {
  const status = execSync("git status --short", { encoding: "utf8" }).trim();
  if (status) {
    console.error("Repo is not clean:");
    console.error(status);
    process.exit(1);
  }
  console.log("Repo is clean.");
} catch (err) {
  console.error("Could not check git status. Is this a git repo?");
  process.exit(1);
}
