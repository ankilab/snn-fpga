/*
------------------------------------------------------------------------------
Module: prediction_selector
Authors: MD & ChatGPT
Date: 16.07.2026

Description:
Determines the final network prediction by computing the argmax over all
class spike counters.

------------------------------------------------------------------------------
*/

module prediction_selector #(
    parameter NUM_CLASSES = 10,
    parameter COUNT_WIDTH = 16,
    parameter CLASS_WIDTH = (NUM_CLASSES <= 1) ? 1 : $clog2(NUM_CLASSES)
) (
    input logic enable,

    // INPUTS
    input logic [COUNT_WIDTH-1:0] class_count [NUM_CLASSES],

    // OUTPUTS
    output logic prediction_valid,
    output logic [CLASS_WIDTH-1:0] prediction
);

    integer i;
    logic [COUNT_WIDTH-1:0] max_count;

    // Use an explicit combinational process for broad simulator support with
    // unpacked array inputs (some Icarus versions warn on always_comb here).
    always @(*) begin
        prediction       = '0;
        prediction_valid = 1'b0;
        max_count        = class_count[0];

        // A prediction is exported only when the decoder completes.
        if (enable) begin
            for (i = 0; i < NUM_CLASSES; i++) begin
                if (class_count[i] != '0)
                    prediction_valid = 1'b1;
            end
        end

        // Argmax (ties -> lowest index)
        for (i = 1; i < NUM_CLASSES; i++) begin
            if (class_count[i] > max_count) begin
                max_count  = class_count[i];
                prediction = CLASS_WIDTH'(i);
            end
        end
    end

endmodule
