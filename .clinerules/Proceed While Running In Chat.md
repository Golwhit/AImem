# Proceed While Running In Chat

## Core Principle
Automate continuation of long-running tasks without requiring manual intervention. The agent should proceed autonomously through multi-step workflows until completion or failure.

## Behavior Guidelines

### Autonomous Execution
- Execute commands sequentially without pausing for user confirmation on routine operations
- Chain dependent operations: when one command completes, immediately begin the next logical step
- Monitor background processes and collect results without requiring explicit approval
- Handle errors gracefully with automatic fallbacks where possible

### Emergency Stop Pattern
- Replace "confirm to proceed" prompts with an **emergency stop** mechanism
- Users can interrupt execution at any time by sending: `KILL ALL`
- On receiving this command, immediately terminate all agent-started processes and their descendants
- Report what was stopped before considering the task complete

### Background Process Management
- Start long-running operations in the background with explicit PID tracking
- Use output redirection to capture logs: `command > /tmp/task.log 2>&1 &`
- Monitor status via periodic checks: `ps aux | grep <pid>` or `tail -f /tmp/task.log`
- Clean up orphaned processes on task completion

### Error Handling
- If a command fails, attempt automatic recovery (retry with adjusted parameters, check prerequisites)
- Only escalate to the user when:
  - Multiple retry attempts have failed
  - The failure requires human decision-making
  - Security-sensitive operations are involved

## Implementation Notes
- Use `nohup` or `&` for background tasks
- Always capture PIDs: `command & echo $! > /tmp/pid.file`
- Set reasonable timeouts to prevent infinite hangs
- Log all autonomous decisions for auditability
