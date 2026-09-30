# Check the actual linked routed checkpoint, not the packaged XO.
open_checkpoint [lindex $argv 0]
report_route_status -file [lindex $argv 1]
report_timing_summary -delay_type min_max -report_unconstrained -file [lindex $argv 2]
set bad_nets [get_nets -hier -filter {ROUTE_STATUS == UNROUTED || ROUTE_STATUS == PARTIAL}]
if {[llength $bad_nets] != 0} { error "Unrouted or partially routed nets remain" }
foreach delay {max min} {
    set paths [get_timing_paths -delay_type $delay -max_paths 1]
    if {[llength $paths] == 0} { error "No $delay timing path found" }
    if {[get_property SLACK [lindex $paths 0]] < 0} { error "$delay timing failed" }
}
puts "ROUTED_SETUP_HOLD_CHECK_PASSED"
close_design
