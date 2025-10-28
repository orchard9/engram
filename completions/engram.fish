# Print an optspec for argparse to handle cmd's options that are independent of any subcommand.
function __fish_engram_global_optspecs
	string join \n l/log-level= h/help V/version
end

function __fish_engram_needs_command
	# Figure out if the current invocation already has a command.
	set -l cmd (commandline -opc)
	set -e cmd[1]
	argparse -s (__fish_engram_global_optspecs) -- $cmd 2>/dev/null
	or return
	if set -q argv[1]
		# Also print the command, so this can be used to figure out what it is.
		echo $argv[1]
		return 1
	end
	return 0
end

function __fish_engram_using_subcommand
	set -l cmd (__fish_engram_needs_command)
	test -z "$cmd"
	and return 1
	contains -- $cmd[1] $argv
end

complete -c engram -n "__fish_engram_needs_command" -s l -l log-level -d 'Set the log level' -r
complete -c engram -n "__fish_engram_needs_command" -s h -l help -d 'Print help'
complete -c engram -n "__fish_engram_needs_command" -s V -l version -d 'Print version'
complete -c engram -n "__fish_engram_needs_command" -f -a "start" -d 'Start the Engram server with automatic configuration'
complete -c engram -n "__fish_engram_needs_command" -f -a "stop" -d 'Stop the Engram server gracefully'
complete -c engram -n "__fish_engram_needs_command" -f -a "status" -d 'Show current status'
complete -c engram -n "__fish_engram_needs_command" -f -a "memory" -d 'Memory operations'
complete -c engram -n "__fish_engram_needs_command" -f -a "space" -d 'Memory space registry operations'
complete -c engram -n "__fish_engram_needs_command" -f -a "config" -d 'Configuration management'
complete -c engram -n "__fish_engram_needs_command" -f -a "shell" -d 'Interactive shell mode'
complete -c engram -n "__fish_engram_needs_command" -f -a "benchmark" -d 'Benchmark server performance'
complete -c engram -n "__fish_engram_needs_command" -f -a "docs" -d 'Show embedded documentation'
complete -c engram -n "__fish_engram_needs_command" -f -a "query" -d 'Query with probabilistic confidence intervals'
complete -c engram -n "__fish_engram_needs_command" -f -a "backup" -d 'Backup operations'
complete -c engram -n "__fish_engram_needs_command" -f -a "restore" -d 'Restore operations'
complete -c engram -n "__fish_engram_needs_command" -f -a "diagnose" -d 'Diagnostic operations'
complete -c engram -n "__fish_engram_needs_command" -f -a "migrate" -d 'Migration operations'
complete -c engram -n "__fish_engram_needs_command" -f -a "validate" -d 'Validation operations'
complete -c engram -n "__fish_engram_needs_command" -f -a "help" -d 'Print this message or the help of the given subcommand(s)'
complete -c engram -n "__fish_engram_using_subcommand start" -s p -l port -d 'Server port (automatically finds free port if default occupied)' -r
complete -c engram -n "__fish_engram_using_subcommand start" -s g -l grpc-port -d 'gRPC server port (automatically finds free port if default occupied)' -r
complete -c engram -n "__fish_engram_using_subcommand start" -s h -l help -d 'Print help'
complete -c engram -n "__fish_engram_using_subcommand stop" -l force -d 'Force shutdown without graceful cleanup'
complete -c engram -n "__fish_engram_using_subcommand stop" -s h -l help -d 'Print help'
complete -c engram -n "__fish_engram_using_subcommand status" -l space -d 'Memory space to query (overrides ENGRAM_MEMORY_SPACE)' -r
complete -c engram -n "__fish_engram_using_subcommand status" -l json -d 'Output in JSON format'
complete -c engram -n "__fish_engram_using_subcommand status" -l watch -d 'Watch status continuously'
complete -c engram -n "__fish_engram_using_subcommand status" -s h -l help -d 'Print help'
complete -c engram -n "__fish_engram_using_subcommand memory; and not __fish_seen_subcommand_from create get search list delete help" -s h -l help -d 'Print help'
complete -c engram -n "__fish_engram_using_subcommand memory; and not __fish_seen_subcommand_from create get search list delete help" -f -a "create" -d 'Create a new memory'
complete -c engram -n "__fish_engram_using_subcommand memory; and not __fish_seen_subcommand_from create get search list delete help" -f -a "get" -d 'Get a memory by ID'
complete -c engram -n "__fish_engram_using_subcommand memory; and not __fish_seen_subcommand_from create get search list delete help" -f -a "search" -d 'Search for memories'
complete -c engram -n "__fish_engram_using_subcommand memory; and not __fish_seen_subcommand_from create get search list delete help" -f -a "list" -d 'List all memories'
complete -c engram -n "__fish_engram_using_subcommand memory; and not __fish_seen_subcommand_from create get search list delete help" -f -a "delete" -d 'Delete a memory by ID'
complete -c engram -n "__fish_engram_using_subcommand memory; and not __fish_seen_subcommand_from create get search list delete help" -f -a "help" -d 'Print this message or the help of the given subcommand(s)'
complete -c engram -n "__fish_engram_using_subcommand memory; and __fish_seen_subcommand_from create" -s c -l confidence -d 'Confidence level (0.0 to 1.0)' -r
complete -c engram -n "__fish_engram_using_subcommand memory; and __fish_seen_subcommand_from create" -l space -d 'Memory space to store in (overrides ENGRAM_MEMORY_SPACE)' -r
complete -c engram -n "__fish_engram_using_subcommand memory; and __fish_seen_subcommand_from create" -s h -l help -d 'Print help'
complete -c engram -n "__fish_engram_using_subcommand memory; and __fish_seen_subcommand_from get" -l space -d 'Memory space to query (overrides ENGRAM_MEMORY_SPACE)' -r
complete -c engram -n "__fish_engram_using_subcommand memory; and __fish_seen_subcommand_from get" -s h -l help -d 'Print help'
complete -c engram -n "__fish_engram_using_subcommand memory; and __fish_seen_subcommand_from search" -s l -l limit -d 'Maximum number of results' -r
complete -c engram -n "__fish_engram_using_subcommand memory; and __fish_seen_subcommand_from search" -l space -d 'Memory space to query (overrides ENGRAM_MEMORY_SPACE)' -r
complete -c engram -n "__fish_engram_using_subcommand memory; and __fish_seen_subcommand_from search" -s h -l help -d 'Print help'
complete -c engram -n "__fish_engram_using_subcommand memory; and __fish_seen_subcommand_from list" -s l -l limit -d 'Maximum number of results' -r
complete -c engram -n "__fish_engram_using_subcommand memory; and __fish_seen_subcommand_from list" -s o -l offset -d 'Skip number of results' -r
complete -c engram -n "__fish_engram_using_subcommand memory; and __fish_seen_subcommand_from list" -s h -l help -d 'Print help'
complete -c engram -n "__fish_engram_using_subcommand memory; and __fish_seen_subcommand_from delete" -s h -l help -d 'Print help'
complete -c engram -n "__fish_engram_using_subcommand memory; and __fish_seen_subcommand_from help" -f -a "create" -d 'Create a new memory'
complete -c engram -n "__fish_engram_using_subcommand memory; and __fish_seen_subcommand_from help" -f -a "get" -d 'Get a memory by ID'
complete -c engram -n "__fish_engram_using_subcommand memory; and __fish_seen_subcommand_from help" -f -a "search" -d 'Search for memories'
complete -c engram -n "__fish_engram_using_subcommand memory; and __fish_seen_subcommand_from help" -f -a "list" -d 'List all memories'
complete -c engram -n "__fish_engram_using_subcommand memory; and __fish_seen_subcommand_from help" -f -a "delete" -d 'Delete a memory by ID'
complete -c engram -n "__fish_engram_using_subcommand memory; and __fish_seen_subcommand_from help" -f -a "help" -d 'Print this message or the help of the given subcommand(s)'
complete -c engram -n "__fish_engram_using_subcommand space; and not __fish_seen_subcommand_from list create help" -s h -l help -d 'Print help'
complete -c engram -n "__fish_engram_using_subcommand space; and not __fish_seen_subcommand_from list create help" -f -a "list" -d 'List all registered memory spaces'
complete -c engram -n "__fish_engram_using_subcommand space; and not __fish_seen_subcommand_from list create help" -f -a "create" -d 'Create (or retrieve) a memory space by identifier'
complete -c engram -n "__fish_engram_using_subcommand space; and not __fish_seen_subcommand_from list create help" -f -a "help" -d 'Print this message or the help of the given subcommand(s)'
complete -c engram -n "__fish_engram_using_subcommand space; and __fish_seen_subcommand_from list" -s h -l help -d 'Print help'
complete -c engram -n "__fish_engram_using_subcommand space; and __fish_seen_subcommand_from create" -s h -l help -d 'Print help'
complete -c engram -n "__fish_engram_using_subcommand space; and __fish_seen_subcommand_from help" -f -a "list" -d 'List all registered memory spaces'
complete -c engram -n "__fish_engram_using_subcommand space; and __fish_seen_subcommand_from help" -f -a "create" -d 'Create (or retrieve) a memory space by identifier'
complete -c engram -n "__fish_engram_using_subcommand space; and __fish_seen_subcommand_from help" -f -a "help" -d 'Print this message or the help of the given subcommand(s)'
complete -c engram -n "__fish_engram_using_subcommand config; and not __fish_seen_subcommand_from get set list path help" -s h -l help -d 'Print help'
complete -c engram -n "__fish_engram_using_subcommand config; and not __fish_seen_subcommand_from get set list path help" -f -a "get" -d 'Get a configuration value'
complete -c engram -n "__fish_engram_using_subcommand config; and not __fish_seen_subcommand_from get set list path help" -f -a "set" -d 'Set a configuration value'
complete -c engram -n "__fish_engram_using_subcommand config; and not __fish_seen_subcommand_from get set list path help" -f -a "list" -d 'Manage configuration settings'
complete -c engram -n "__fish_engram_using_subcommand config; and not __fish_seen_subcommand_from get set list path help" -f -a "path" -d 'Show configuration file location'
complete -c engram -n "__fish_engram_using_subcommand config; and not __fish_seen_subcommand_from get set list path help" -f -a "help" -d 'Print this message or the help of the given subcommand(s)'
complete -c engram -n "__fish_engram_using_subcommand config; and __fish_seen_subcommand_from get" -s h -l help -d 'Print help'
complete -c engram -n "__fish_engram_using_subcommand config; and __fish_seen_subcommand_from set" -s h -l help -d 'Print help'
complete -c engram -n "__fish_engram_using_subcommand config; and __fish_seen_subcommand_from list" -l section -d 'Show only specified section' -r
complete -c engram -n "__fish_engram_using_subcommand config; and __fish_seen_subcommand_from list" -s h -l help -d 'Print help'
complete -c engram -n "__fish_engram_using_subcommand config; and __fish_seen_subcommand_from path" -s h -l help -d 'Print help'
complete -c engram -n "__fish_engram_using_subcommand config; and __fish_seen_subcommand_from help" -f -a "get" -d 'Get a configuration value'
complete -c engram -n "__fish_engram_using_subcommand config; and __fish_seen_subcommand_from help" -f -a "set" -d 'Set a configuration value'
complete -c engram -n "__fish_engram_using_subcommand config; and __fish_seen_subcommand_from help" -f -a "list" -d 'Manage configuration settings'
complete -c engram -n "__fish_engram_using_subcommand config; and __fish_seen_subcommand_from help" -f -a "path" -d 'Show configuration file location'
complete -c engram -n "__fish_engram_using_subcommand config; and __fish_seen_subcommand_from help" -f -a "help" -d 'Print this message or the help of the given subcommand(s)'
complete -c engram -n "__fish_engram_using_subcommand shell" -s h -l help -d 'Print help'
complete -c engram -n "__fish_engram_using_subcommand benchmark; and not __fish_seen_subcommand_from latency throughput spreading consolidation help" -s h -l help -d 'Print help'
complete -c engram -n "__fish_engram_using_subcommand benchmark; and not __fish_seen_subcommand_from latency throughput spreading consolidation help" -f -a "latency" -d 'Measure operation latency (P50, P95, P99)'
complete -c engram -n "__fish_engram_using_subcommand benchmark; and not __fish_seen_subcommand_from latency throughput spreading consolidation help" -f -a "throughput" -d 'Measure throughput (operations per second)'
complete -c engram -n "__fish_engram_using_subcommand benchmark; and not __fish_seen_subcommand_from latency throughput spreading consolidation help" -f -a "spreading" -d 'Benchmark spreading activation performance'
complete -c engram -n "__fish_engram_using_subcommand benchmark; and not __fish_seen_subcommand_from latency throughput spreading consolidation help" -f -a "consolidation" -d 'Benchmark memory consolidation'
complete -c engram -n "__fish_engram_using_subcommand benchmark; and not __fish_seen_subcommand_from latency throughput spreading consolidation help" -f -a "help" -d 'Print this message or the help of the given subcommand(s)'
complete -c engram -n "__fish_engram_using_subcommand benchmark; and __fish_seen_subcommand_from latency" -s i -l iterations -d 'Number of iterations' -r
complete -c engram -n "__fish_engram_using_subcommand benchmark; and __fish_seen_subcommand_from latency" -s w -l warmup -d 'Warmup iterations' -r
complete -c engram -n "__fish_engram_using_subcommand benchmark; and __fish_seen_subcommand_from latency" -s h -l help -d 'Print help'
complete -c engram -n "__fish_engram_using_subcommand benchmark; and __fish_seen_subcommand_from throughput" -s d -l duration -d 'Duration to run benchmark (seconds)' -r
complete -c engram -n "__fish_engram_using_subcommand benchmark; and __fish_seen_subcommand_from throughput" -s c -l clients -d 'Number of concurrent clients' -r
complete -c engram -n "__fish_engram_using_subcommand benchmark; and __fish_seen_subcommand_from throughput" -s h -l help -d 'Print help'
complete -c engram -n "__fish_engram_using_subcommand benchmark; and __fish_seen_subcommand_from spreading" -s n -l nodes -d 'Number of nodes to activate' -r
complete -c engram -n "__fish_engram_using_subcommand benchmark; and __fish_seen_subcommand_from spreading" -s d -l depth -d 'Activation spread depth' -r
complete -c engram -n "__fish_engram_using_subcommand benchmark; and __fish_seen_subcommand_from spreading" -s h -l help -d 'Print help'
complete -c engram -n "__fish_engram_using_subcommand benchmark; and __fish_seen_subcommand_from consolidation" -s l -l load-test -d 'Simulate consolidation load'
complete -c engram -n "__fish_engram_using_subcommand benchmark; and __fish_seen_subcommand_from consolidation" -s h -l help -d 'Print help'
complete -c engram -n "__fish_engram_using_subcommand benchmark; and __fish_seen_subcommand_from help" -f -a "latency" -d 'Measure operation latency (P50, P95, P99)'
complete -c engram -n "__fish_engram_using_subcommand benchmark; and __fish_seen_subcommand_from help" -f -a "throughput" -d 'Measure throughput (operations per second)'
complete -c engram -n "__fish_engram_using_subcommand benchmark; and __fish_seen_subcommand_from help" -f -a "spreading" -d 'Benchmark spreading activation performance'
complete -c engram -n "__fish_engram_using_subcommand benchmark; and __fish_seen_subcommand_from help" -f -a "consolidation" -d 'Benchmark memory consolidation'
complete -c engram -n "__fish_engram_using_subcommand benchmark; and __fish_seen_subcommand_from help" -f -a "help" -d 'Print this message or the help of the given subcommand(s)'
complete -c engram -n "__fish_engram_using_subcommand docs" -l export -d 'Export documentation to file' -r
complete -c engram -n "__fish_engram_using_subcommand docs" -l list -d 'Show all available sections'
complete -c engram -n "__fish_engram_using_subcommand docs" -s h -l help -d 'Print help'
complete -c engram -n "__fish_engram_using_subcommand query" -s l -l limit -d 'Maximum number of results' -r
complete -c engram -n "__fish_engram_using_subcommand query" -s f -l format -d 'Output format (json, table, compact)' -r
complete -c engram -n "__fish_engram_using_subcommand query" -l space -d 'Memory space to query (overrides ENGRAM_MEMORY_SPACE)' -r
complete -c engram -n "__fish_engram_using_subcommand query" -s h -l help -d 'Print help'
complete -c engram -n "__fish_engram_using_subcommand backup; and not __fish_seen_subcommand_from create list verify prune help" -s h -l help -d 'Print help'
complete -c engram -n "__fish_engram_using_subcommand backup; and not __fish_seen_subcommand_from create list verify prune help" -f -a "create" -d 'Create a new backup'
complete -c engram -n "__fish_engram_using_subcommand backup; and not __fish_seen_subcommand_from create list verify prune help" -f -a "list" -d 'List available backups'
complete -c engram -n "__fish_engram_using_subcommand backup; and not __fish_seen_subcommand_from create list verify prune help" -f -a "verify" -d 'Verify backup integrity'
complete -c engram -n "__fish_engram_using_subcommand backup; and not __fish_seen_subcommand_from create list verify prune help" -f -a "prune" -d 'Prune old backups according to retention policy'
complete -c engram -n "__fish_engram_using_subcommand backup; and not __fish_seen_subcommand_from create list verify prune help" -f -a "help" -d 'Print this message or the help of the given subcommand(s)'
complete -c engram -n "__fish_engram_using_subcommand backup; and __fish_seen_subcommand_from create" -s t -l backup-type -d 'Backup type: full or incremental' -r
complete -c engram -n "__fish_engram_using_subcommand backup; and __fish_seen_subcommand_from create" -s s -l space -d 'Memory space to backup (or "all")' -r
complete -c engram -n "__fish_engram_using_subcommand backup; and __fish_seen_subcommand_from create" -s o -l output -d 'Output directory for backup' -r -F
complete -c engram -n "__fish_engram_using_subcommand backup; and __fish_seen_subcommand_from create" -s c -l compression -d 'Compression level (1-9, default: 3)' -r
complete -c engram -n "__fish_engram_using_subcommand backup; and __fish_seen_subcommand_from create" -l progress -d 'Show progress bar'
complete -c engram -n "__fish_engram_using_subcommand backup; and __fish_seen_subcommand_from create" -s h -l help -d 'Print help'
complete -c engram -n "__fish_engram_using_subcommand backup; and __fish_seen_subcommand_from list" -s t -l backup-type -d 'Filter by backup type' -r
complete -c engram -n "__fish_engram_using_subcommand backup; and __fish_seen_subcommand_from list" -s s -l space -d 'Filter by memory space' -r
complete -c engram -n "__fish_engram_using_subcommand backup; and __fish_seen_subcommand_from list" -s f -l format -d 'Output format' -r
complete -c engram -n "__fish_engram_using_subcommand backup; and __fish_seen_subcommand_from list" -s h -l help -d 'Print help'
complete -c engram -n "__fish_engram_using_subcommand backup; and __fish_seen_subcommand_from verify" -s l -l level -d 'Verification level: L1 (manifest), L2 (checksums), L3 (structure), L4 (full restore test)' -r
complete -c engram -n "__fish_engram_using_subcommand backup; and __fish_seen_subcommand_from verify" -s v -l verbose -d 'Show detailed output'
complete -c engram -n "__fish_engram_using_subcommand backup; and __fish_seen_subcommand_from verify" -s h -l help -d 'Print help'
complete -c engram -n "__fish_engram_using_subcommand backup; and __fish_seen_subcommand_from prune" -l daily -d 'Retention: daily backups to keep' -r
complete -c engram -n "__fish_engram_using_subcommand backup; and __fish_seen_subcommand_from prune" -l weekly -d 'Retention: weekly backups to keep' -r
complete -c engram -n "__fish_engram_using_subcommand backup; and __fish_seen_subcommand_from prune" -l monthly -d 'Retention: monthly backups to keep' -r
complete -c engram -n "__fish_engram_using_subcommand backup; and __fish_seen_subcommand_from prune" -l dry-run -d 'Dry run (show what would be deleted without deleting)'
complete -c engram -n "__fish_engram_using_subcommand backup; and __fish_seen_subcommand_from prune" -s y -l yes -d 'Confirm deletion without prompt'
complete -c engram -n "__fish_engram_using_subcommand backup; and __fish_seen_subcommand_from prune" -s h -l help -d 'Print help'
complete -c engram -n "__fish_engram_using_subcommand backup; and __fish_seen_subcommand_from help" -f -a "create" -d 'Create a new backup'
complete -c engram -n "__fish_engram_using_subcommand backup; and __fish_seen_subcommand_from help" -f -a "list" -d 'List available backups'
complete -c engram -n "__fish_engram_using_subcommand backup; and __fish_seen_subcommand_from help" -f -a "verify" -d 'Verify backup integrity'
complete -c engram -n "__fish_engram_using_subcommand backup; and __fish_seen_subcommand_from help" -f -a "prune" -d 'Prune old backups according to retention policy'
complete -c engram -n "__fish_engram_using_subcommand backup; and __fish_seen_subcommand_from help" -f -a "help" -d 'Print this message or the help of the given subcommand(s)'
complete -c engram -n "__fish_engram_using_subcommand restore; and not __fish_seen_subcommand_from full incremental pitr verify-only help" -s h -l help -d 'Print help'
complete -c engram -n "__fish_engram_using_subcommand restore; and not __fish_seen_subcommand_from full incremental pitr verify-only help" -f -a "full" -d 'Restore from full backup'
complete -c engram -n "__fish_engram_using_subcommand restore; and not __fish_seen_subcommand_from full incremental pitr verify-only help" -f -a "incremental" -d 'Apply incremental backup'
complete -c engram -n "__fish_engram_using_subcommand restore; and not __fish_seen_subcommand_from full incremental pitr verify-only help" -f -a "pitr" -d 'Point-in-time recovery'
complete -c engram -n "__fish_engram_using_subcommand restore; and not __fish_seen_subcommand_from full incremental pitr verify-only help" -f -a "verify-only" -d 'Verify restore without applying'
complete -c engram -n "__fish_engram_using_subcommand restore; and not __fish_seen_subcommand_from full incremental pitr verify-only help" -f -a "help" -d 'Print this message or the help of the given subcommand(s)'
complete -c engram -n "__fish_engram_using_subcommand restore; and __fish_seen_subcommand_from full" -s t -l target -d 'Target directory for restore' -r -F
complete -c engram -n "__fish_engram_using_subcommand restore; and __fish_seen_subcommand_from full" -l progress -d 'Show progress bar'
complete -c engram -n "__fish_engram_using_subcommand restore; and __fish_seen_subcommand_from full" -s h -l help -d 'Print help'
complete -c engram -n "__fish_engram_using_subcommand restore; and __fish_seen_subcommand_from incremental" -l progress -d 'Show progress bar'
complete -c engram -n "__fish_engram_using_subcommand restore; and __fish_seen_subcommand_from incremental" -s h -l help -d 'Print help'
complete -c engram -n "__fish_engram_using_subcommand restore; and __fish_seen_subcommand_from pitr" -s t -l target -d 'Target directory for restore' -r -F
complete -c engram -n "__fish_engram_using_subcommand restore; and __fish_seen_subcommand_from pitr" -s h -l help -d 'Print help'
complete -c engram -n "__fish_engram_using_subcommand restore; and __fish_seen_subcommand_from verify-only" -s h -l help -d 'Print help'
complete -c engram -n "__fish_engram_using_subcommand restore; and __fish_seen_subcommand_from help" -f -a "full" -d 'Restore from full backup'
complete -c engram -n "__fish_engram_using_subcommand restore; and __fish_seen_subcommand_from help" -f -a "incremental" -d 'Apply incremental backup'
complete -c engram -n "__fish_engram_using_subcommand restore; and __fish_seen_subcommand_from help" -f -a "pitr" -d 'Point-in-time recovery'
complete -c engram -n "__fish_engram_using_subcommand restore; and __fish_seen_subcommand_from help" -f -a "verify-only" -d 'Verify restore without applying'
complete -c engram -n "__fish_engram_using_subcommand restore; and __fish_seen_subcommand_from help" -f -a "help" -d 'Print this message or the help of the given subcommand(s)'
complete -c engram -n "__fish_engram_using_subcommand diagnose; and not __fish_seen_subcommand_from health collect analyze-logs emergency help" -s h -l help -d 'Print help'
complete -c engram -n "__fish_engram_using_subcommand diagnose; and not __fish_seen_subcommand_from health collect analyze-logs emergency help" -f -a "health" -d 'Run comprehensive health check'
complete -c engram -n "__fish_engram_using_subcommand diagnose; and not __fish_seen_subcommand_from health collect analyze-logs emergency help" -f -a "collect" -d 'Collect debug bundle for support'
complete -c engram -n "__fish_engram_using_subcommand diagnose; and not __fish_seen_subcommand_from health collect analyze-logs emergency help" -f -a "analyze-logs" -d 'Analyze logs for patterns and errors'
complete -c engram -n "__fish_engram_using_subcommand diagnose; and not __fish_seen_subcommand_from health collect analyze-logs emergency help" -f -a "emergency" -d 'Emergency recovery procedures'
complete -c engram -n "__fish_engram_using_subcommand diagnose; and not __fish_seen_subcommand_from health collect analyze-logs emergency help" -f -a "help" -d 'Print this message or the help of the given subcommand(s)'
complete -c engram -n "__fish_engram_using_subcommand diagnose; and __fish_seen_subcommand_from health" -s o -l output -d 'Output file for report' -r -F
complete -c engram -n "__fish_engram_using_subcommand diagnose; and __fish_seen_subcommand_from health" -l strict -d 'Fail with non-zero exit code on warnings'
complete -c engram -n "__fish_engram_using_subcommand diagnose; and __fish_seen_subcommand_from health" -s h -l help -d 'Print help'
complete -c engram -n "__fish_engram_using_subcommand diagnose; and __fish_seen_subcommand_from collect" -l log-lines -d 'Include full logs (last N lines)' -r
complete -c engram -n "__fish_engram_using_subcommand diagnose; and __fish_seen_subcommand_from collect" -l include-dumps -d 'Include memory dumps'
complete -c engram -n "__fish_engram_using_subcommand diagnose; and __fish_seen_subcommand_from collect" -s h -l help -d 'Print help'
complete -c engram -n "__fish_engram_using_subcommand diagnose; and __fish_seen_subcommand_from analyze-logs" -s f -l file -d 'Log file path (default: system logs)' -r -F
complete -c engram -n "__fish_engram_using_subcommand diagnose; and __fish_seen_subcommand_from analyze-logs" -s w -l window -d 'Time window: 1h, 24h, 7d' -r
complete -c engram -n "__fish_engram_using_subcommand diagnose; and __fish_seen_subcommand_from analyze-logs" -s s -l severity -d 'Filter by severity: ERROR, WARN, INFO' -r
complete -c engram -n "__fish_engram_using_subcommand diagnose; and __fish_seen_subcommand_from analyze-logs" -s h -l help -d 'Print help'
complete -c engram -n "__fish_engram_using_subcommand diagnose; and __fish_seen_subcommand_from emergency" -l auto -d 'Automatic recovery without prompts'
complete -c engram -n "__fish_engram_using_subcommand diagnose; and __fish_seen_subcommand_from emergency" -s h -l help -d 'Print help'
complete -c engram -n "__fish_engram_using_subcommand diagnose; and __fish_seen_subcommand_from help" -f -a "health" -d 'Run comprehensive health check'
complete -c engram -n "__fish_engram_using_subcommand diagnose; and __fish_seen_subcommand_from help" -f -a "collect" -d 'Collect debug bundle for support'
complete -c engram -n "__fish_engram_using_subcommand diagnose; and __fish_seen_subcommand_from help" -f -a "analyze-logs" -d 'Analyze logs for patterns and errors'
complete -c engram -n "__fish_engram_using_subcommand diagnose; and __fish_seen_subcommand_from help" -f -a "emergency" -d 'Emergency recovery procedures'
complete -c engram -n "__fish_engram_using_subcommand diagnose; and __fish_seen_subcommand_from help" -f -a "help" -d 'Print this message or the help of the given subcommand(s)'
complete -c engram -n "__fish_engram_using_subcommand migrate; and not __fish_seen_subcommand_from neo4j postgresql redis help" -s h -l help -d 'Print help'
complete -c engram -n "__fish_engram_using_subcommand migrate; and not __fish_seen_subcommand_from neo4j postgresql redis help" -f -a "neo4j" -d 'Migrate from Neo4j'
complete -c engram -n "__fish_engram_using_subcommand migrate; and not __fish_seen_subcommand_from neo4j postgresql redis help" -f -a "postgresql" -d 'Migrate from PostgreSQL'
complete -c engram -n "__fish_engram_using_subcommand migrate; and not __fish_seen_subcommand_from neo4j postgresql redis help" -f -a "redis" -d 'Migrate from Redis'
complete -c engram -n "__fish_engram_using_subcommand migrate; and not __fish_seen_subcommand_from neo4j postgresql redis help" -f -a "help" -d 'Print this message or the help of the given subcommand(s)'
complete -c engram -n "__fish_engram_using_subcommand migrate; and __fish_seen_subcommand_from neo4j" -s t -l target-space -d 'Target memory space' -r
complete -c engram -n "__fish_engram_using_subcommand migrate; and __fish_seen_subcommand_from neo4j" -s b -l batch-size -d 'Batch size for migration' -r
complete -c engram -n "__fish_engram_using_subcommand migrate; and __fish_seen_subcommand_from neo4j" -s h -l help -d 'Print help'
complete -c engram -n "__fish_engram_using_subcommand migrate; and __fish_seen_subcommand_from postgresql" -s t -l target-space -d 'Target memory space' -r
complete -c engram -n "__fish_engram_using_subcommand migrate; and __fish_seen_subcommand_from postgresql" -s m -l mappings -d 'Table mappings configuration file' -r -F
complete -c engram -n "__fish_engram_using_subcommand migrate; and __fish_seen_subcommand_from postgresql" -s h -l help -d 'Print help'
complete -c engram -n "__fish_engram_using_subcommand migrate; and __fish_seen_subcommand_from redis" -s t -l target-space -d 'Target memory space' -r
complete -c engram -n "__fish_engram_using_subcommand migrate; and __fish_seen_subcommand_from redis" -s k -l key-pattern -d 'Key pattern to migrate' -r
complete -c engram -n "__fish_engram_using_subcommand migrate; and __fish_seen_subcommand_from redis" -s h -l help -d 'Print help'
complete -c engram -n "__fish_engram_using_subcommand migrate; and __fish_seen_subcommand_from help" -f -a "neo4j" -d 'Migrate from Neo4j'
complete -c engram -n "__fish_engram_using_subcommand migrate; and __fish_seen_subcommand_from help" -f -a "postgresql" -d 'Migrate from PostgreSQL'
complete -c engram -n "__fish_engram_using_subcommand migrate; and __fish_seen_subcommand_from help" -f -a "redis" -d 'Migrate from Redis'
complete -c engram -n "__fish_engram_using_subcommand migrate; and __fish_seen_subcommand_from help" -f -a "help" -d 'Print this message or the help of the given subcommand(s)'
complete -c engram -n "__fish_engram_using_subcommand validate; and not __fish_seen_subcommand_from config data deployment help" -s h -l help -d 'Print help'
complete -c engram -n "__fish_engram_using_subcommand validate; and not __fish_seen_subcommand_from config data deployment help" -f -a "config" -d 'Validate configuration file'
complete -c engram -n "__fish_engram_using_subcommand validate; and not __fish_seen_subcommand_from config data deployment help" -f -a "data" -d 'Validate data integrity'
complete -c engram -n "__fish_engram_using_subcommand validate; and not __fish_seen_subcommand_from config data deployment help" -f -a "deployment" -d 'Pre-deployment validation checklist'
complete -c engram -n "__fish_engram_using_subcommand validate; and not __fish_seen_subcommand_from config data deployment help" -f -a "help" -d 'Print this message or the help of the given subcommand(s)'
complete -c engram -n "__fish_engram_using_subcommand validate; and __fish_seen_subcommand_from config" -s f -l file -d 'Path to config file' -r -F
complete -c engram -n "__fish_engram_using_subcommand validate; and __fish_seen_subcommand_from config" -s d -l deployment -d 'Check deployment-specific settings' -r
complete -c engram -n "__fish_engram_using_subcommand validate; and __fish_seen_subcommand_from config" -s h -l help -d 'Print help'
complete -c engram -n "__fish_engram_using_subcommand validate; and __fish_seen_subcommand_from data" -s s -l space -d 'Memory space to validate' -r
complete -c engram -n "__fish_engram_using_subcommand validate; and __fish_seen_subcommand_from data" -l fix -d 'Fix issues automatically'
complete -c engram -n "__fish_engram_using_subcommand validate; and __fish_seen_subcommand_from data" -s h -l help -d 'Print help'
complete -c engram -n "__fish_engram_using_subcommand validate; and __fish_seen_subcommand_from deployment" -s e -l environment -d 'Target environment: dev, staging, production' -r
complete -c engram -n "__fish_engram_using_subcommand validate; and __fish_seen_subcommand_from deployment" -s h -l help -d 'Print help'
complete -c engram -n "__fish_engram_using_subcommand validate; and __fish_seen_subcommand_from help" -f -a "config" -d 'Validate configuration file'
complete -c engram -n "__fish_engram_using_subcommand validate; and __fish_seen_subcommand_from help" -f -a "data" -d 'Validate data integrity'
complete -c engram -n "__fish_engram_using_subcommand validate; and __fish_seen_subcommand_from help" -f -a "deployment" -d 'Pre-deployment validation checklist'
complete -c engram -n "__fish_engram_using_subcommand validate; and __fish_seen_subcommand_from help" -f -a "help" -d 'Print this message or the help of the given subcommand(s)'
complete -c engram -n "__fish_engram_using_subcommand help; and not __fish_seen_subcommand_from start stop status memory space config shell benchmark docs query backup restore diagnose migrate validate help" -f -a "start" -d 'Start the Engram server with automatic configuration'
complete -c engram -n "__fish_engram_using_subcommand help; and not __fish_seen_subcommand_from start stop status memory space config shell benchmark docs query backup restore diagnose migrate validate help" -f -a "stop" -d 'Stop the Engram server gracefully'
complete -c engram -n "__fish_engram_using_subcommand help; and not __fish_seen_subcommand_from start stop status memory space config shell benchmark docs query backup restore diagnose migrate validate help" -f -a "status" -d 'Show current status'
complete -c engram -n "__fish_engram_using_subcommand help; and not __fish_seen_subcommand_from start stop status memory space config shell benchmark docs query backup restore diagnose migrate validate help" -f -a "memory" -d 'Memory operations'
complete -c engram -n "__fish_engram_using_subcommand help; and not __fish_seen_subcommand_from start stop status memory space config shell benchmark docs query backup restore diagnose migrate validate help" -f -a "space" -d 'Memory space registry operations'
complete -c engram -n "__fish_engram_using_subcommand help; and not __fish_seen_subcommand_from start stop status memory space config shell benchmark docs query backup restore diagnose migrate validate help" -f -a "config" -d 'Configuration management'
complete -c engram -n "__fish_engram_using_subcommand help; and not __fish_seen_subcommand_from start stop status memory space config shell benchmark docs query backup restore diagnose migrate validate help" -f -a "shell" -d 'Interactive shell mode'
complete -c engram -n "__fish_engram_using_subcommand help; and not __fish_seen_subcommand_from start stop status memory space config shell benchmark docs query backup restore diagnose migrate validate help" -f -a "benchmark" -d 'Benchmark server performance'
complete -c engram -n "__fish_engram_using_subcommand help; and not __fish_seen_subcommand_from start stop status memory space config shell benchmark docs query backup restore diagnose migrate validate help" -f -a "docs" -d 'Show embedded documentation'
complete -c engram -n "__fish_engram_using_subcommand help; and not __fish_seen_subcommand_from start stop status memory space config shell benchmark docs query backup restore diagnose migrate validate help" -f -a "query" -d 'Query with probabilistic confidence intervals'
complete -c engram -n "__fish_engram_using_subcommand help; and not __fish_seen_subcommand_from start stop status memory space config shell benchmark docs query backup restore diagnose migrate validate help" -f -a "backup" -d 'Backup operations'
complete -c engram -n "__fish_engram_using_subcommand help; and not __fish_seen_subcommand_from start stop status memory space config shell benchmark docs query backup restore diagnose migrate validate help" -f -a "restore" -d 'Restore operations'
complete -c engram -n "__fish_engram_using_subcommand help; and not __fish_seen_subcommand_from start stop status memory space config shell benchmark docs query backup restore diagnose migrate validate help" -f -a "diagnose" -d 'Diagnostic operations'
complete -c engram -n "__fish_engram_using_subcommand help; and not __fish_seen_subcommand_from start stop status memory space config shell benchmark docs query backup restore diagnose migrate validate help" -f -a "migrate" -d 'Migration operations'
complete -c engram -n "__fish_engram_using_subcommand help; and not __fish_seen_subcommand_from start stop status memory space config shell benchmark docs query backup restore diagnose migrate validate help" -f -a "validate" -d 'Validation operations'
complete -c engram -n "__fish_engram_using_subcommand help; and not __fish_seen_subcommand_from start stop status memory space config shell benchmark docs query backup restore diagnose migrate validate help" -f -a "help" -d 'Print this message or the help of the given subcommand(s)'
complete -c engram -n "__fish_engram_using_subcommand help; and __fish_seen_subcommand_from memory" -f -a "create" -d 'Create a new memory'
complete -c engram -n "__fish_engram_using_subcommand help; and __fish_seen_subcommand_from memory" -f -a "get" -d 'Get a memory by ID'
complete -c engram -n "__fish_engram_using_subcommand help; and __fish_seen_subcommand_from memory" -f -a "search" -d 'Search for memories'
complete -c engram -n "__fish_engram_using_subcommand help; and __fish_seen_subcommand_from memory" -f -a "list" -d 'List all memories'
complete -c engram -n "__fish_engram_using_subcommand help; and __fish_seen_subcommand_from memory" -f -a "delete" -d 'Delete a memory by ID'
complete -c engram -n "__fish_engram_using_subcommand help; and __fish_seen_subcommand_from space" -f -a "list" -d 'List all registered memory spaces'
complete -c engram -n "__fish_engram_using_subcommand help; and __fish_seen_subcommand_from space" -f -a "create" -d 'Create (or retrieve) a memory space by identifier'
complete -c engram -n "__fish_engram_using_subcommand help; and __fish_seen_subcommand_from config" -f -a "get" -d 'Get a configuration value'
complete -c engram -n "__fish_engram_using_subcommand help; and __fish_seen_subcommand_from config" -f -a "set" -d 'Set a configuration value'
complete -c engram -n "__fish_engram_using_subcommand help; and __fish_seen_subcommand_from config" -f -a "list" -d 'Manage configuration settings'
complete -c engram -n "__fish_engram_using_subcommand help; and __fish_seen_subcommand_from config" -f -a "path" -d 'Show configuration file location'
complete -c engram -n "__fish_engram_using_subcommand help; and __fish_seen_subcommand_from benchmark" -f -a "latency" -d 'Measure operation latency (P50, P95, P99)'
complete -c engram -n "__fish_engram_using_subcommand help; and __fish_seen_subcommand_from benchmark" -f -a "throughput" -d 'Measure throughput (operations per second)'
complete -c engram -n "__fish_engram_using_subcommand help; and __fish_seen_subcommand_from benchmark" -f -a "spreading" -d 'Benchmark spreading activation performance'
complete -c engram -n "__fish_engram_using_subcommand help; and __fish_seen_subcommand_from benchmark" -f -a "consolidation" -d 'Benchmark memory consolidation'
complete -c engram -n "__fish_engram_using_subcommand help; and __fish_seen_subcommand_from backup" -f -a "create" -d 'Create a new backup'
complete -c engram -n "__fish_engram_using_subcommand help; and __fish_seen_subcommand_from backup" -f -a "list" -d 'List available backups'
complete -c engram -n "__fish_engram_using_subcommand help; and __fish_seen_subcommand_from backup" -f -a "verify" -d 'Verify backup integrity'
complete -c engram -n "__fish_engram_using_subcommand help; and __fish_seen_subcommand_from backup" -f -a "prune" -d 'Prune old backups according to retention policy'
complete -c engram -n "__fish_engram_using_subcommand help; and __fish_seen_subcommand_from restore" -f -a "full" -d 'Restore from full backup'
complete -c engram -n "__fish_engram_using_subcommand help; and __fish_seen_subcommand_from restore" -f -a "incremental" -d 'Apply incremental backup'
complete -c engram -n "__fish_engram_using_subcommand help; and __fish_seen_subcommand_from restore" -f -a "pitr" -d 'Point-in-time recovery'
complete -c engram -n "__fish_engram_using_subcommand help; and __fish_seen_subcommand_from restore" -f -a "verify-only" -d 'Verify restore without applying'
complete -c engram -n "__fish_engram_using_subcommand help; and __fish_seen_subcommand_from diagnose" -f -a "health" -d 'Run comprehensive health check'
complete -c engram -n "__fish_engram_using_subcommand help; and __fish_seen_subcommand_from diagnose" -f -a "collect" -d 'Collect debug bundle for support'
complete -c engram -n "__fish_engram_using_subcommand help; and __fish_seen_subcommand_from diagnose" -f -a "analyze-logs" -d 'Analyze logs for patterns and errors'
complete -c engram -n "__fish_engram_using_subcommand help; and __fish_seen_subcommand_from diagnose" -f -a "emergency" -d 'Emergency recovery procedures'
complete -c engram -n "__fish_engram_using_subcommand help; and __fish_seen_subcommand_from migrate" -f -a "neo4j" -d 'Migrate from Neo4j'
complete -c engram -n "__fish_engram_using_subcommand help; and __fish_seen_subcommand_from migrate" -f -a "postgresql" -d 'Migrate from PostgreSQL'
complete -c engram -n "__fish_engram_using_subcommand help; and __fish_seen_subcommand_from migrate" -f -a "redis" -d 'Migrate from Redis'
complete -c engram -n "__fish_engram_using_subcommand help; and __fish_seen_subcommand_from validate" -f -a "config" -d 'Validate configuration file'
complete -c engram -n "__fish_engram_using_subcommand help; and __fish_seen_subcommand_from validate" -f -a "data" -d 'Validate data integrity'
complete -c engram -n "__fish_engram_using_subcommand help; and __fish_seen_subcommand_from validate" -f -a "deployment" -d 'Pre-deployment validation checklist'
