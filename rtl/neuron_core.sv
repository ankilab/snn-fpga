/*
------------------------------------------------------------------------------
Module: neuron_core

Description:
Model-independent neuron update interface. NEURON_MODEL is a synthesis-time
selection with identifiers shared by python/snn_reference/model.py:
    0 = LIF
    1 = Izhikevich
    2 = Hodgkin-Huxley

All implementations must obey the start/done handshake and emit a one-cycle
spike pulse. This wrapper keeps the population and layer control independent
of the latency and state vector of a concrete neuron model.
------------------------------------------------------------------------------
*/

module neuron_core #(
    parameter int unsigned NEURON_MODEL = 0,
    parameter int unsigned MEM_WIDTH = 24,
    parameter int unsigned CURRENT_WIDTH = 24,
    parameter logic signed [MEM_WIDTH-1:0] THRESHOLD = 100,
    parameter logic signed [MEM_WIDTH-1:0] RESET_POT = 0,
    parameter int unsigned LEAK_SHIFT = 2,
    parameter int unsigned IZH_FRAC_BITS = 8,
    parameter int unsigned IZH_A_NUMERATOR = 1,
    parameter int unsigned IZH_A_DENOMINATOR = 50,
    parameter int unsigned IZH_B_NUMERATOR = 1,
    parameter int unsigned IZH_B_DENOMINATOR = 5,
    parameter integer IZH_RESET_VOLTAGE = -65,
    parameter integer IZH_RECOVERY_INCREMENT = 8,
    parameter integer IZH_SPIKE_THRESHOLD = 30,
    parameter HH_LUT_DIR = "data/hh_luts/"
) (
    input logic clk,
    input logic rst,
    input logic clear,
    input logic start,
    input logic signed [CURRENT_WIDTH-1:0] syn_current,
    output logic done,
    output logic spike,
    output logic signed [MEM_WIDTH-1:0] membrane
);
    localparam int unsigned NEURON_MODEL_LIF = 0;
    localparam int unsigned NEURON_MODEL_IZHIKEVICH = 1;
    localparam int unsigned NEURON_MODEL_HODGKIN_HUXLEY = 2;

    generate
        if (NEURON_MODEL == NEURON_MODEL_LIF) begin : gen_lif
            lif_neuron #(
                .MEM_WIDTH(MEM_WIDTH),
                .CURRENT_WIDTH(CURRENT_WIDTH),
                .THRESHOLD(THRESHOLD),
                .RESET_POT(RESET_POT),
                .LEAK_SHIFT(LEAK_SHIFT)
            ) u_lif_neuron (
                .clk(clk),
                .rst(rst),
                .clear(clear),
                .state_load(1'b0),
                .state_membrane('0),
                .start(start),
                .syn_current(syn_current),
                .done(done),
                .spike(spike),
                .membrane(membrane)
            );
        end else if (NEURON_MODEL == NEURON_MODEL_IZHIKEVICH) begin : gen_izhikevich
            izhikevich_neuron #(
                .MEM_WIDTH(MEM_WIDTH),
                .CURRENT_WIDTH(CURRENT_WIDTH),
                .FRAC_BITS(IZH_FRAC_BITS),
                .A_NUMERATOR(IZH_A_NUMERATOR),
                .A_DENOMINATOR(IZH_A_DENOMINATOR),
                .B_NUMERATOR(IZH_B_NUMERATOR),
                .B_DENOMINATOR(IZH_B_DENOMINATOR),
                .RESET_VOLTAGE(IZH_RESET_VOLTAGE),
                .RECOVERY_INCREMENT(IZH_RECOVERY_INCREMENT),
                .SPIKE_THRESHOLD(IZH_SPIKE_THRESHOLD)
            ) u_izhikevich_neuron (
                .clk(clk),
                .rst(rst),
                .clear(clear),
                .state_load(1'b0),
                .state_voltage('0),
                .state_recovery('0),
                .start(start),
                .syn_current(syn_current),
                .done(done),
                .spike(spike),
                .membrane(membrane),
                .recovery()
            );
        end else if (NEURON_MODEL == NEURON_MODEL_HODGKIN_HUXLEY) begin : gen_hh
            hodgkin_huxley_neuron #(
                .MEM_WIDTH(MEM_WIDTH),
                .CURRENT_WIDTH(CURRENT_WIDTH),
                .LUT_DIR(HH_LUT_DIR)
            ) u_hodgkin_huxley_neuron (
                .clk(clk),
                .rst(rst),
                .clear(clear),
                .state_load(1'b0),
                .state_voltage('0),
                .state_sodium_activation('0),
                .state_sodium_inactivation('0),
                .state_potassium_activation('0),
                .start(start),
                .syn_current(syn_current),
                .done(done),
                .spike(spike),
                .membrane(membrane),
                .sodium_activation(),
                .sodium_inactivation(),
                .potassium_activation()
            );
        end else begin : gen_unsupported
            initial $fatal(1, "Unsupported NEURON_MODEL=%0d", NEURON_MODEL);
            assign done = 1'b0;
            assign spike = 1'b0;
            assign membrane = '0;
        end
    endgenerate
endmodule
