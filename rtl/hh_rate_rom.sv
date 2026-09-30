/*
------------------------------------------------------------------------------
Module: hh_rate_rom

Description:
Synchronous Hodgkin-Huxley rate lookup table. Vivado uses an explicit XPM
single-port block ROM; portable RTL simulation uses the equivalent readmemh
model with one clock cycle of latency.
------------------------------------------------------------------------------
*/

module hh_rate_rom #(
    parameter INIT_FILE = ""
) (
    input  logic        clk,
    input  logic [9:0]  address,
    output logic [23:0] data
);
`ifdef SYNTHESIS
    wire [23:0] xpm_data;

    xpm_memory_sprom #(
        .ADDR_WIDTH_A(10),
        .AUTO_SLEEP_TIME(0),
        .CASCADE_HEIGHT(0),
        .ECC_MODE("no_ecc"),
        .MEMORY_INIT_FILE(INIT_FILE),
        .MEMORY_INIT_PARAM(""),
        .MEMORY_OPTIMIZATION("true"),
        .MEMORY_PRIMITIVE("block"),
        .MEMORY_SIZE(641 * 24),
        .MESSAGE_CONTROL(0),
        .READ_DATA_WIDTH_A(24),
        .READ_LATENCY_A(1),
        .READ_RESET_VALUE_A("0"),
        .RST_MODE_A("SYNC"),
        .SIM_ASSERT_CHK(0),
        .USE_MEM_INIT(1),
        .WAKEUP_TIME("disable_sleep")
    ) rate_rom (
        .dbiterra(),
        .douta(xpm_data),
        .sbiterra(),
        .addra(address),
        .clka(clk),
        .ena(1'b1),
        .injectdbiterra(1'b0),
        .injectsbiterra(1'b0),
        .regcea(1'b1),
        .rsta(1'b0),
        .sleep(1'b0)
    );

    always_comb data = xpm_data;
`else
    logic [23:0] memory [0:640];

    initial $readmemh(INIT_FILE, memory);

    always_ff @(posedge clk)
        data <= memory[address];
`endif
endmodule
