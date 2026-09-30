# Package the existing AXI4-Lite accelerators as user-managed Vitis RTL kernels.

set script_dir [file dirname [file normalize [info script]]]
set repo_root [file normalize [file join $script_dir .. ..]]

proc usage {} {
    puts {Usage: vivado -mode batch -source vivado/u50/package_u50_kernel.tcl -tclargs ?options?}
    puts {  -kernel snn                 Accelerator to package (default: snn)}
    puts {  -output PATH                    Output .xo path}
    puts {  -build_dir PATH                 Generated output root}
    puts {  -part PART                      Packaging part (default: xcu50-fsvh2104-2-e)}
    puts {  -hidden_neuron_model 0|1|2      SNN hidden LIF/IZH/HH (default: 0)}
    puts {  -output_neuron_model 0|1|2      SNN output LIF/IZH/HH (default: 0)}
    puts {  -input_neurons N                Input neurons (default: 784)}
    puts {  -hidden_neurons N               Hidden neurons (default: 256)}
    puts {  -output_neurons N               Output neurons (default: 10)}
    puts {  -num_events N                   SNN event capacity (default: 12544)}
    puts {  -hidden_executor_count N         Hidden executors; 0 selects model default (default: 0)}
    puts {  -output_executor_count N         Output executors; 0 selects model default (default: 0)}
    puts {  -interface_minor 0|4|5|6         Interface minor; 0 selects 6 for explicit executors, else 4}
}

proc parse_args {arguments repo_root} {
    set options [dict create kernel snn output "" build_dir "" \
        part xcu50-fsvh2104-2-e hidden_neuron_model 0 output_neuron_model 0 \
        input_neurons 784 hidden_neurons 256 output_neurons 10 num_events 12544 \
        hidden_executor_count 0 output_executor_count 0 interface_minor 0]
    for {set i 0} {$i < [llength $arguments]} {incr i} {
        set key [lindex $arguments $i]
        if {$key in {-h -help --help}} { usage; return "" }
        if {$key ni {-kernel -output -build_dir -part -hidden_neuron_model \
                -output_neuron_model -input_neurons -hidden_neurons \
                -output_neurons -num_events -hidden_executor_count \
                -output_executor_count -interface_minor}} {
            error "Unknown option '$key'"
        }
        incr i
        if {$i >= [llength $arguments]} { error "Missing value after '$key'" }
        dict set options [string range $key 1 end] [lindex $arguments $i]
    }
    if {[dict get $options kernel] ne "snn"} { error "-kernel must be snn" }
    foreach key {input_neurons hidden_neurons output_neurons num_events} {
        set value [dict get $options $key]
        if {![string is integer -strict $value] || $value < 1 || $value > 65535} {
            error "-$key must be an integer from 1 through 65535"
        }
    }
    foreach key {hidden_neuron_model output_neuron_model} {
        if {[dict get $options $key] ni {0 1 2}} { error "-$key must be 0, 1, or 2" }
    }
    foreach key {hidden_executor_count output_executor_count} {
        set value [dict get $options $key]
        if {![string is integer -strict $value] || $value < 0 || $value > 65535} {
            error "-$key must be an integer from 0 through 65535"
        }
    }
    if {[dict get $options interface_minor] ni {0 4 5 6}} {
        error "-interface_minor must be 0, 4, 5, or 6"
    }
    if {[dict get $options build_dir] eq ""} {
        dict set options build_dir [file join $repo_root build u50 [dict get $options kernel]]
    }
    dict set options build_dir [file normalize [dict get $options build_dir]]
    if {[dict get $options output] eq ""} {
        dict set options output [file join [dict get $options build_dir] artifacts \
            "[dict get $options kernel]_accelerator.xo"]
    }
    dict set options output [file normalize [dict get $options output]]
    return $options
}

proc set_parameter_default {core name value} {
    set user_parameter [ipx::get_user_parameters -quiet $name -of_objects $core]
    if {[llength $user_parameter] == 1} { set_property value $value $user_parameter }
    set hdl_parameter [ipx::get_hdl_parameters -quiet $name -of_objects $core]
    if {[llength $hdl_parameter] == 1} { set_property value $value $hdl_parameter }
}

proc set_parameter_fixed {core name value} {
    set user_parameter [ipx::get_user_parameters -quiet $name -of_objects $core]
    if {[llength $user_parameter] == 1} {
        set_property value_dependency {} $user_parameter
        set_property value $value $user_parameter
    }
    set hdl_parameter [ipx::get_hdl_parameters -quiet $name -of_objects $core]
    if {[llength $hdl_parameter] == 1} {
        set_property value_dependency {} $hdl_parameter
        set_property value $value $hdl_parameter
    }
}

# Vitis 2026.1 derives xd:register metadata from concrete IP-XACT registers.
# The user-managed kernel XML reserves offset 0xF8, so represent that same
# placeholder in the packaged AXI4-Lite memory map.
proc ensure_control_register_metadata {core} {
    set memory_map [ipx::get_memory_maps -quiet S_AXI -of_objects $core]
    if {[llength $memory_map] != 1} { error "Expected one packaged S_AXI memory map" }
    set address_block [ipx::get_address_blocks -quiet Reg -of_objects $memory_map]
    if {[llength $address_block] == 0} {
        set address_block [ipx::add_address_block Reg $memory_map]
    } else {
        set address_block [lindex $address_block 0]
    }
    set_property range 65536 $address_block
    set_property width 32 $address_block
    set_property usage register $address_block
    set register [ipx::get_registers -quiet reserved -of_objects $address_block]
    if {[llength $register] == 0} {
        set register [ipx::add_register reserved $address_block]
    } else {
        set register [lindex $register 0]
    }
    set_property address_offset 248 $register
    set_property size 32 $register
    set_property access read-write $register
}

proc mark_as_vitis_kernel {ip_root options} {
    set edit_dir [file join [dict get $options build_dir] kernel_edit]
    create_project -force u50_kernel_edit $edit_dir -part [dict get $options part]
    ipx::edit_ip_in_project -upgrade true -name kernel_edit \
        -directory $edit_dir [file join $ip_root component.xml]
    set core [ipx::current_core]

    set control [ipx::get_bus_interfaces -quiet S_AXI -of_objects $core]
    if {[llength $control] != 1} { error "Expected one packaged S_AXI interface" }
    set_property name S_AXI_CONTROL $control

    set clock [ipx::get_bus_interfaces -quiet S_AXI_CLK -of_objects $core]
    if {[llength $clock] == 1} {
        set_property name ap_clk $clock
        set associated_busif [ipx::get_bus_parameters -quiet ASSOCIATED_BUSIF -of_objects $clock]
        if {[llength $associated_busif] == 1} {
            set_property value S_AXI_CONTROL $associated_busif
        }
    }
    set reset [ipx::get_bus_interfaces -quiet S_AXI_RST -of_objects $core]
    if {[llength $reset] == 1} { set_property name ap_rst_n $reset }
    set interrupt [ipx::get_bus_interfaces -quiet IRQ -of_objects $core]
    if {[llength $interrupt] == 1} { set_property name interrupt $interrupt }
    ensure_control_register_metadata $core

    foreach key {INPUT_NEURONS HIDDEN_NEURONS OUTPUT_NEURONS} option \
            {input_neurons hidden_neurons output_neurons} {
        set_parameter_default $core $key [dict get $options $option]
    }
    if {[dict get $options kernel] eq "snn"} {
        foreach key {NUM_EVENTS HIDDEN_NEURON_MODEL OUTPUT_NEURON_MODEL} option \
                {num_events hidden_neuron_model output_neuron_model} {
            set_parameter_default $core $key [dict get $options $option]
        }
        set hidden_executor_count [dict get $options hidden_executor_count]
        if {$hidden_executor_count == 0} {
            set hidden_executor_count [expr {
                [dict get $options hidden_neuron_model] == 0
                    ? [dict get $options hidden_neurons] : 1
            }]
        }
        set output_executor_count [dict get $options output_executor_count]
        if {$output_executor_count == 0} {
            set output_executor_count [expr {
                [dict get $options output_neuron_model] == 0
                    ? [dict get $options output_neurons] : 1
            }]
        }
        if {$hidden_executor_count > [dict get $options hidden_neurons]} {
            error "-hidden_executor_count exceeds -hidden_neurons"
        }
        if {$output_executor_count > [dict get $options output_neurons]} {
            error "-output_executor_count exceeds -output_neurons"
        }
        set interface_minor [dict get $options interface_minor]
        if {$interface_minor == 0} {
            set explicit_executor_architecture [expr {
                [dict get $options hidden_executor_count] != 0 ||
                [dict get $options output_executor_count] != 0
            }]
            set interface_minor [expr {$explicit_executor_architecture ? 6 : 4}]
        }
        set interface_version [expr {0x00010000 | $interface_minor}]
        # These HDL parameters have conditional SystemVerilog defaults. Remove
        # that dependency in the packaged IP so Vivado does not regenerate the
        # full-parallel LIF values when this release requests 1/1 executors.
        set_parameter_fixed $core HIDDEN_EXECUTOR_COUNT $hidden_executor_count
        set_parameter_fixed $core OUTPUT_EXECUTOR_COUNT $output_executor_count
        set_parameter_default $core INTERFACE_VERSION $interface_version
    }

    set_property sdx_kernel true $core
    set_property sdx_kernel_type rtl $core
    ipx::update_checksums $core
    ipx::check_integrity -kernel $core
    ipx::save_core $core
    close_project
}

proc main {arguments repo_root} {
    set options [parse_args $arguments $repo_root]
    if {$options eq ""} { return }
    set build_root [dict get $options build_dir]
    file mkdir $build_root [file dirname [dict get $options output]]
    set ::env(SNN_REPO_ROOT) $repo_root

    source [file join $repo_root vivado common.tcl]
    set ip_root [snn_package_ip [dict get $options part] $build_root]
    set kernel_name snn_accelerator
    mark_as_vitis_kernel $ip_root $options
    # On Windows, package_xo can fail to replace an existing archive even with
    # -force. Remove only the exact generated output after validating its path.
    if {[file exists [dict get $options output]]} {
        file delete -force [dict get $options output]
    }
    package_xo -force -ctrl_protocol user_managed \
        -xo_path [dict get $options output] \
        -kernel_name $kernel_name -ip_directory $ip_root \
        -kernel_xml [file join $repo_root vivado u50 \
            "[dict get $options kernel]_kernel.xml"]
    if {![file isfile [dict get $options output]]} { error "XO was not generated" }
    puts "INFO: U50 RTL kernel written to [dict get $options output]"
}

if {[catch {main $argv $repo_root} message options]} {
    puts stderr "ERROR: $message"
    if {[dict exists $options -errorinfo]} { puts stderr [dict get $options -errorinfo] }
    exit 1
}
exit 0
