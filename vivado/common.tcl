# Shared SNN source list and Vitis RTL-kernel packaging helpers.

if {[info exists ::env(SNN_REPO_ROOT)] && $::env(SNN_REPO_ROOT) ne ""} {
    set ::snn_repo_root [file normalize $::env(SNN_REPO_ROOT)]
    set ::snn_script_dir [file join $::snn_repo_root vivado]
} else {
    set ::snn_script_dir [file dirname [file normalize [info script]]]
    set ::snn_repo_root [file normalize [file join $::snn_script_dir ..]]
}
set ::snn_base_address 0xA0000000
set ::snn_address_range 0x00010000
set ::snn_ip_vlnv "snn.dev:snn:snn_accelerator:1.0"

set ::snn_rtl_sources [list \
    [file join $::snn_repo_root rtl lif_neuron.sv] \
    [file join $::snn_repo_root rtl izhikevich_neuron.sv] \
    [file join $::snn_repo_root rtl hh_rate_rom.sv] \
    [file join $::snn_repo_root rtl signed_round_divider.sv] \
    [file join $::snn_repo_root rtl hodgkin_huxley_neuron.sv] \
    [file join $::snn_repo_root rtl neuron_core.sv] \
    [file join $::snn_repo_root rtl neuron_array.sv] \
    [file join $::snn_repo_root rtl output_decoder.sv] \
    [file join $::snn_repo_root rtl output_interface.sv] \
    [file join $::snn_repo_root rtl prediction_selector.sv] \
    [file join $::snn_repo_root rtl snn_controller.sv] \
    [file join $::snn_repo_root rtl snn_layer.sv] \
    [file join $::snn_repo_root rtl spike_event_encoder.sv] \
    [file join $::snn_repo_root rtl spike_memory.sv] \
    [file join $::snn_repo_root rtl spike_reader.sv] \
    [file join $::snn_repo_root rtl spike_router.sv] \
    [file join $::snn_repo_root rtl synapse_accumulator.sv] \
    [file join $::snn_repo_root rtl synapse_bram.sv] \
    [file join $::snn_repo_root rtl weight_controller.sv] \
    [file join $::snn_repo_root rtl snn_top.sv] \
    [file join $::snn_repo_root rtl snn_axi_lite_wrapper.sv] \
]

set ::snn_hh_lut_sources [glob -nocomplain \
    [file join $::snn_repo_root data hh_luts hh_*.mem] \
]

proc snn_require_sources {} {
    foreach source_file $::snn_rtl_sources {
        if {![file isfile $source_file]} {
            error "Required RTL source is missing: $source_file"
        }
    }
    if {[llength $::snn_hh_lut_sources] != 6} {
        error "Expected six Hodgkin-Huxley LUT files under data/hh_luts"
    }
}

proc snn_assert_generated_path {path build_root} {
    set normalized_path [file normalize $path]
    set normalized_root [file normalize $build_root]
    set ancestor [file dirname $normalized_path]
    while {![string equal -nocase $ancestor [file dirname $ancestor]]} {
        if {[string equal -nocase $ancestor $normalized_root]} {
            return
        }
        set ancestor [file dirname $ancestor]
    }
    error "Refusing to replace generated path outside build root: $normalized_path"
}

proc snn_package_ip {part_name build_root} {
    snn_require_sources

    set pack_project_dir [file join $build_root ip_pack_project]
    set ip_root [file join $build_root ip_repo snn_accelerator_1_0]
    snn_assert_generated_path $pack_project_dir $build_root
    snn_assert_generated_path $ip_root $build_root
    file delete -force $pack_project_dir $ip_root
    file mkdir [file dirname $ip_root]

    create_project -force snn_ip_pack $pack_project_dir -part $part_name
    add_files -norecurse $::snn_rtl_sources
    add_files -norecurse $::snn_hh_lut_sources
    set_property top snn_axi_lite_wrapper [current_fileset]
    update_compile_order -fileset sources_1

    ipx::package_project \
        -root_dir $ip_root \
        -vendor snn.dev \
        -library snn \
        -taxonomy /UserIP \
        -import_files \
        -set_current true

    set core [ipx::current_core]
    set_property name snn_accelerator $core
    set_property display_name {SNN AXI4-Lite Accelerator} $core
    set_property description {Two-layer configurable-neuron SNN inference accelerator} $core
    set_property vendor_display_name {SNN Project} $core
    set_property version 1.0 $core
    set_property core_revision 1 $core

    ipx::infer_bus_interfaces xilinx.com:interface:aximm_rtl:1.0 $core
    ipx::infer_bus_interfaces xilinx.com:signal:clock_rtl:1.0 $core
    ipx::infer_bus_interfaces xilinx.com:signal:reset_rtl:1.0 $core
    ipx::infer_bus_interfaces xilinx.com:signal:interrupt_rtl:1.0 $core

    set s_axi [ipx::get_bus_interfaces -quiet S_AXI -of_objects $core]
    if {[llength $s_axi] != 1} {
        error "IP packager did not infer exactly one S_AXI interface. Check X_INTERFACE metadata."
    }

    set memory_map [ipx::get_memory_maps -quiet S_AXI -of_objects $core]
    if {[llength $memory_map] == 0} {
        set memory_map [ipx::add_memory_map S_AXI $core]
    } else {
        set memory_map [lindex $memory_map 0]
    }
    set address_block [ipx::get_address_blocks -quiet Reg -of_objects $memory_map]
    if {[llength $address_block] == 0} {
        set address_block [ipx::add_address_block Reg $memory_map]
    } else {
        set address_block [lindex $address_block 0]
    }
    set_property range $::snn_address_range $address_block
    set_property width 32 $address_block
    set_property usage register $address_block
    set_property slave_memory_map_ref S_AXI $s_axi

    ipx::create_xgui_files $core
    ipx::update_checksums $core
    ipx::check_integrity $core
    ipx::save_core $core
    close_project
    return $ip_root
}
