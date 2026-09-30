set part_name xcu50-fsvh2104-2-e
set matches [get_parts -quiet $part_name]
if {[llength $matches] != 1} {
    puts stderr "ERROR: Vivado device part '$part_name' is not installed"
    exit 1
}
puts "INFO: Vivado device part is installed: [lindex $matches 0]"
exit 0
