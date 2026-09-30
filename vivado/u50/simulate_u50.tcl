# Behavioral XSim regression for the RTL reused by the U50 kernels.
set script_dir [file dirname [file normalize [info script]]]
set repo_root [file normalize [file join $script_dir .. ..]]
set kernel snn
set part xcu50-fsvh2104-2-e
set project_dir ""
for {set i 0} {$i < [llength $argv]} {incr i} {
    set key [lindex $argv $i]
    if {$key ni {-kernel -part -build_dir}} {
        error {Usage: vivado -mode batch -source vivado/u50/simulate_u50.tcl -tclargs ?-kernel snn? ?-part PART? ?-build_dir PATH?}
    }
    incr i
    if {$i >= [llength $argv]} { error "Missing value after $key" }
    if {$key eq "-kernel"} { set kernel [lindex $argv $i] }
    if {$key eq "-part"} { set part [lindex $argv $i] }
    if {$key eq "-build_dir"} { set project_dir [file normalize [lindex $argv $i]] }
}
if {$kernel ne "snn"} { error "Kernel must be snn" }
if {$project_dir eq ""} { set project_dir [file join $repo_root build u50 simulation $kernel] }
create_project -force "u50_${kernel}_simulation" $project_dir -part $part
set_property target_simulator XSim [current_project]
source [file join $repo_root vivado common.tcl]
snn_require_sources
add_files -norecurse $::snn_rtl_sources
add_files -norecurse $::snn_hh_lut_sources
add_files -fileset sim_1 -norecurse [file join $repo_root tb tb_snn_axi_lite_wrapper.sv]
set_property top tb_snn_axi_lite_wrapper [get_filesets sim_1]
update_compile_order -fileset sources_1
update_compile_order -fileset sim_1
launch_simulation -simset sim_1 -mode behavioral
run all
close_sim
close_project
puts "INFO: U50 $kernel AXI-Lite behavioral simulation passed"
exit 0
