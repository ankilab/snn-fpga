set parts [get_parts -quiet *xcu50*]
set boards [get_board_parts -quiet *u50*]
puts "U50_INSPECTION_BEGIN"
puts "XCU50_PARTS=[join $parts ,]"
puts "U50_BOARDS=[join $boards ,]"
puts "U50_INSPECTION_END"
exit [expr {([llength $parts] == 0) ? 2 : 0}]
