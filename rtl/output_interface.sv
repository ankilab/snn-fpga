/*
------------------------------------------------------------------------------
Module: output_interface

Description:
Exports the final classification result to external logic or software
through a configurable communication interface (e.g., AXI-Stream, UART,
GPIO, or PCIe).

------------------------------------------------------------------------------
*/

module output_interface #(
    parameter CLASS_WIDTH = 4
) (
    // INPUTS
    input logic clk,
    input logic rst,

    input logic prediction_valid,
    input logic [CLASS_WIDTH-1:0] prediction,

    // OUTPUTS
    output logic tx_valid,
    output logic [CLASS_WIDTH-1:0] tx_data
);

    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            tx_valid <= 1'b0;
            tx_data  <= '0;
        end
        else begin
            tx_valid <= prediction_valid;

            if (prediction_valid) begin
                tx_data <= prediction;
            end
        end
    end

endmodule
