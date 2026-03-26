# Claude Code Configuration — Nodeglass

## Implementation Quality

- Before writing code, copy every AC checkbox and every behavioral constraint from the Linear issue body into a comment block at the top of the working file. Treat the full issue body as spec, not just the checkboxes.
- When a spec table contains exact values (risk weights, field names, enum members), copy them directly from the source of truth. Never retype from memory.
- If anything in the issue description conflicts with or extends the AC checkboxes, surface it before coding. Do not silently drop requirements.
- After implementation, walk every AC line and every behavioral statement in the issue description against the code before marking the issue "Done." If a criterion cannot be verified from the diff alone, write a test that proves it.
- Linear is the single source of truth for acceptance criteria. If the plan file disagrees with Linear, Linear wins.

## File Organization

- Never save working files, text, markdown, or tests to the root folder.
- `/src` — source code
- `/tests` — test files
- `/docs` — documentation and markdown (only when explicitly requested)
- `/config` — configuration files
- `/scripts` — utility scripts
- `/work-logs` — review reports, retrospectives, execution logs
- `/examples` — example code

## Build & Test

- Always run tests after making code changes.
- Always verify the build succeeds before committing.
- Always read a file before editing it.
- Always use python3 when invoking python from command line
- 

## Security

- Never hardcode API keys, secrets, or credentials in source files.
- Never commit `.env` files or any file containing secrets.
- Validate user input at system boundaries.
