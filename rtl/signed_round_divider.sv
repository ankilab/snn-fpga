/*
------------------------------------------------------------------------------
Module: signed_round_divider

Description:
Iterative signed division by a positive constant. ROUND_NEAREST selects
round-to-nearest with ties away from zero; otherwise the result truncates
toward zero. One quotient bit is produced per cycle, keeping the critical path
to one small compare/subtract stage.
------------------------------------------------------------------------------
*/

module signed_round_divider #(
    parameter int unsigned WIDTH = 40,
    parameter int unsigned DIVISOR = 100,
    parameter bit ROUND_NEAREST = 1'b1
) (
    input  logic                    clk,
    input  logic                    rst,
    input  logic                    start,
    input  logic signed [WIDTH-1:0] value,
    output logic                    done,
    output logic signed [WIDTH-1:0] quotient
);
    localparam int unsigned REM_WIDTH = $clog2(DIVISOR);
    localparam int unsigned COUNT_WIDTH = $clog2(WIDTH);
    localparam logic [REM_WIDTH:0] DIVISOR_TRIAL =
        (REM_WIDTH + 1)'(DIVISOR);
    localparam logic [WIDTH-1:0] HALF_DIVISOR = WIDTH'(DIVISOR / 2);
    localparam logic [COUNT_WIDTH-1:0] LAST_BIT = COUNT_WIDTH'(WIDTH - 1);

    typedef enum logic [1:0] {IDLE, DIVIDE, APPLY_SIGN} state_t;
    state_t state;

    logic negative_reg;
    logic [WIDTH-1:0] dividend_reg;
    logic [WIDTH-1:0] quotient_reg;
    logic [REM_WIDTH-1:0] remainder_reg;
    logic [COUNT_WIDTH-1:0] bit_count;
    logic [REM_WIDTH:0] shifted_remainder;
    logic [REM_WIDTH-1:0] reduced_remainder;
    logic quotient_bit;
    logic [WIDTH-1:0] shifted_quotient;
    logic [WIDTH-1:0] magnitude;

    always_comb begin
        magnitude = value[WIDTH-1] ? (~value + 1'b1) : value;
        shifted_remainder = {remainder_reg, dividend_reg[WIDTH-1]};
        quotient_bit = shifted_remainder >= DIVISOR_TRIAL;
        reduced_remainder = REM_WIDTH'(quotient_bit
            ? shifted_remainder - DIVISOR_TRIAL
            : shifted_remainder);
        shifted_quotient = {quotient_reg[WIDTH-2:0], quotient_bit};
    end

    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            state <= IDLE;
            negative_reg <= 1'b0;
            dividend_reg <= 0;
            quotient_reg <= 0;
            remainder_reg <= 0;
            bit_count <= 0;
            quotient <= 0;
            done <= 1'b0;
        end else begin
            done <= 1'b0;
            case (state)
                IDLE: begin
                    if (start) begin
                        negative_reg <= value[WIDTH-1];
                        dividend_reg <= magnitude
                            + (ROUND_NEAREST ? HALF_DIVISOR : '0);
                        quotient_reg <= 0;
                        remainder_reg <= 0;
                        bit_count <= 0;
                        state <= DIVIDE;
                    end
                end
                DIVIDE: begin
                    dividend_reg <= {dividend_reg[WIDTH-2:0], 1'b0};
                    quotient_reg <= shifted_quotient;
                    remainder_reg <= reduced_remainder;
                    if (bit_count == LAST_BIT) begin
                        state <= APPLY_SIGN;
                    end else begin
                        bit_count <= bit_count + 1'b1;
                    end
                end
                APPLY_SIGN: begin
                    quotient <= negative_reg
                        ? -$signed(quotient_reg)
                        : $signed(quotient_reg);
                    done <= 1'b1;
                    state <= IDLE;
                end
                default: state <= IDLE;
            endcase
        end
    end
endmodule
