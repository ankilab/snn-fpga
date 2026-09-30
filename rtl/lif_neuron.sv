/*
------------------------------------------------------------------------------
Module: lif_neuron
Authors: MD & ChatGPT
Date: 22.07.2026

Description:
Discrete-time Leaky Integrate-and-Fire neuron. 

Handshake:
    start : capture synaptic current and begin update
    done  : asserted when neuron update is complete

Update equation:

    V[t+1] = V[t] - V[t]/2^LEAK_SHIFT + I_syn

Note: This version includes a FSM (IDLE, UPDATE, DONE), which
    can be adapted for more complex neurons that might
    need more than one cycle for an update.

------------------------------------------------------------------------------
*/

module lif_neuron #(
    parameter MEM_WIDTH      = 24,
    parameter CURRENT_WIDTH  = 24,
    parameter logic signed [MEM_WIDTH-1:0] THRESHOLD = 100,
    parameter logic signed [MEM_WIDTH-1:0] RESET_POT = 0,
    parameter LEAK_SHIFT    = 2
)(
    // INPUTS
    input logic clk,
    input logic rst,
    input logic clear,

    // Control handshake. state_load restores externally stored state for a
    // time-multiplexed population; parallel instances keep it deasserted.
    input logic state_load,
    input logic signed [MEM_WIDTH-1:0] state_membrane,
    input logic start,

    // Input current for this update event
    input logic signed [CURRENT_WIDTH-1:0] syn_current,

    // OUTPUTS
    // Completion signal
    output logic done,

    // Outputs
    output logic spike,
    output logic signed [MEM_WIDTH-1:0] membrane
);
    // IMPORTANT: MEM_WIDTH == CURRENT_WIDTH
    initial begin
        if (MEM_WIDTH != CURRENT_WIDTH)
            $error("lif_neuron requires MEM_WIDTH == CURRENT_WIDTH");
    end

    //----------------------------------------------------------------------
    // Internal state
    //----------------------------------------------------------------------
    typedef enum logic [1:0] {
        IDLE,
        UPDATE,
        DONE
    } state_t;

    state_t state, next_state;

    //----------------------------------------------------------------------
    // Latched input current
    //----------------------------------------------------------------------
    logic signed [CURRENT_WIDTH-1:0] current_reg;

    //----------------------------------------------------------------------
    // Intermediate calculation
    //----------------------------------------------------------------------
    logic signed [MEM_WIDTH-1:0] leak_term;
    logic signed [MEM_WIDTH-1:0] next_mem;

    always_comb begin
        leak_term = membrane >>> LEAK_SHIFT;
        next_mem = membrane - leak_term + current_reg;
    end

    //----------------------------------------------------------------------
    // FSM
    //----------------------------------------------------------------------
    always_ff @(posedge clk or posedge rst) begin
        if (rst)
            state <= IDLE;
        else if (clear)
            state <= IDLE;
        else
            state <= next_state;
    end

    // NEXT-STATE LOGIC
    always_comb begin
        next_state = state;
        case(state)
            IDLE:
            begin
                if(start)
                    next_state = UPDATE;
            end

            UPDATE:
            begin
                next_state = DONE;
            end

            DONE:
            begin
                next_state = IDLE;
            end

            default:
                next_state = IDLE;
        endcase
    end

    //----------------------------------------------------------------------
    // Datapath (Output Logic)
    //----------------------------------------------------------------------
    
    always_ff @(posedge clk or posedge rst) begin
        if(rst || clear) begin
            membrane    <= RESET_POT;
            current_reg <= '0;
            spike <= 1'b0;
            done  <= 1'b0;
        end

        else if (state_load) begin
            membrane <= state_membrane;
            current_reg <= '0;
            spike <= 1'b0;
            done <= 1'b0;
        end

        else begin
            // defaults
            spike <= 1'b0;
            done  <= 1'b0;
            case(state)
                IDLE:
                begin
                    if(start) begin
                        // Freeze input for this update
                        current_reg <= syn_current;
                    end
                end

                UPDATE:
                begin
                    if(next_mem >= THRESHOLD) begin
                        membrane <= RESET_POT;
                        spike    <= 1'b1;
                    end
                    else begin
                        membrane <= next_mem;
                    end
                end

                DONE:
                begin
                    done <= 1'b1;
                end

                default:
                begin
                end
            endcase
        end
    end

endmodule
