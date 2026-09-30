// Compatibility replacement for the legacy U50 platform's HDL-encrypted
// clk_metadata_adapter. Its block-design Tcl propagates the clock metadata;
// the synthesized hardware is a transparent clock connection.
module clk_metadata_adapter_v1_0_0 (
    input  wire clk_in,
    output wire clk_out
);
    assign clk_out = clk_in;
endmodule
