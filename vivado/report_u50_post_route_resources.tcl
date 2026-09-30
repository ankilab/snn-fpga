# Extract reproducible utilization, hierarchy, clock and timing evidence from a
# routed U50 Vitis checkpoint.
#
# Usage:
#   vivado -mode batch -source vivado/report_u50_post_route_resources.tcl \
#     -tclargs <routed.dcp> <output-directory>

if {$argc != 2} {
    error "Usage: report_u50_post_route_resources.tcl <routed.dcp> <output-directory>"
}

set checkpoint [file normalize [lindex $argv 0]]
set output_dir [file normalize [lindex $argv 1]]
if {![file isfile $checkpoint]} {
    error "Routed checkpoint does not exist: $checkpoint"
}
file mkdir $output_dir

open_checkpoint $checkpoint
set route_status [get_property ROUTE_STATUS [current_design]]

report_route_status \
    -file [file join $output_dir post_route_status.rpt]
report_utilization \
    -file [file join $output_dir post_route_utilization.rpt]
report_utilization \
    -hierarchical \
    -hierarchical_depth 12 \
    -file [file join $output_dir post_route_utilization_hierarchical.rpt]
report_timing_summary \
    -delay_type min_max \
    -max_paths 20 \
    -report_unconstrained \
    -file [file join $output_dir post_route_timing_summary.rpt]
report_clock_utilization \
    -file [file join $output_dir post_route_clock_utilization.rpt]

# Preserve the non-primitive hierarchy so category boundaries can be audited
# independently of the human-readable utilization report.
set inventory [open [file join $output_dir hierarchy.tsv] w]
puts $inventory "cell\tref_name"
foreach cell [lsort [get_cells -hierarchical -filter {IS_PRIMITIVE == 0}]] {
    puts $inventory "$cell\t[get_property REF_NAME $cell]"
}
close $inventory

set metadata [open [file join $output_dir extraction_metadata.txt] w]
puts $metadata "schema=snn-u50-post-route-extraction-v1"
puts $metadata "checkpoint=$checkpoint"
puts $metadata "checkpoint_bytes=[file size $checkpoint]"
puts $metadata "design=[current_design]"
puts $metadata "part=[get_property PART [current_design]]"
puts $metadata "design_mode=[get_property DESIGN_MODE [current_design]]"
puts $metadata "route_status=$route_status"
puts $metadata "vivado_version=[version -short]"
close $metadata

close_design
puts "U50_POST_ROUTE_RESOURCE_EXTRACTION_COMPLETE output_dir=$output_dir"
