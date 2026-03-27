# cuda-git-commit-miner

By changing only `GIT_COMMITTER_NAME` while keeping the commit message stable, you can mine any commit hash prefix you want (bottleneck: compute). The suggested amend command restores the original message text without adding a mined date suffix. The final mined commit must be unsigned because a signature changes the commit object and therefore the hash, but a signed `HEAD` is still fine: the miner ignores the existing `gpgsig` header and the suggested amend command uses `--no-gpg-sign` only for that amend, so you can keep your normal Git signing defaults unchanged. Most repos use SHA1, but it is possible to toggle SHA256. This script only works for SHA1 and will terminate if it detects that the repo is configured to use SHA256. CUDA is preferred, but actually this script has a CPU mining fallback. When targeting 7 specific digits on a base model M4 mac mini it takes less than 30 sec to CPU mine. When targeting 7 digits on an RTX 4070 ti using CUDA you instantly find the solution.

## Build

* `make`

## Usage

Create any commit, then run:

* `./gitminer-head` (if you want 7 leading zeroes). Then just run the suggested command.

or if you want a custom commit hash prefix run

* `./gitminer-head --prefix 1234567`

Other targets could be readable words, like:

* `/gitminer-head --prefix 0facade`
* `/gitminer-head --prefix 1decade`
* `/gitminer-head --prefix beeeeef`
* `/gitminer-head --prefix faceb00`
* `/gitminer-head --prefix caffeee`
