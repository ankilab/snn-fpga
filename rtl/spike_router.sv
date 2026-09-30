/*
------------------------------------------------------------------------------
Module: spike_router
Authors: MD & ChatGPT
Date: 16.07.2026

Description:
Routes valid spike events to the synaptic subsystem by converting neuron
identifiers into requests for synaptic weight lookups.

Currently, the router is implemented as a simple pass-through:
    neuron_id -> pre_idx

Future extensions may include:
- Hierarchical routing
- Crossbar or NoC support
- Neuron partitioning across cores
- Event filtering and masking
------------------------------------------------------------------------------
*/

module spike_router #(
    parameter int NEURON_ID_WIDTH = 8
) (
    // INPUTS
    input logic event_valid,
    input logic [NEURON_ID_WIDTH-1:0] neuron_id,

    // OUTPUTS
    output logic route_valid,
    output logic [NEURON_ID_WIDTH-1:0] pre_idx
);

    always_comb begin
        route_valid = event_valid;
        pre_idx     = neuron_id;
    end

endmodule
