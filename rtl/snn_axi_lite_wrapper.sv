/* AXI4-Lite control/configuration wrapper for the fixed two-layer SNN. */
module snn_axi_lite_wrapper #(
    parameter int unsigned AXI_ADDR_WIDTH = 8,
    parameter int unsigned NUM_EVENTS = 784,
    parameter int unsigned INPUT_NEURONS = 784,
    parameter int unsigned HIDDEN_NEURONS = 256,
    parameter int unsigned OUTPUT_NEURONS = 10,
    parameter int unsigned TIME_WIDTH = 32,
    parameter int unsigned MEM_WIDTH = 24,
    parameter int unsigned CURRENT_WIDTH = 24,
    parameter int unsigned WEIGHT_WIDTH = 8,
    parameter int unsigned COUNT_WIDTH = 16,
    parameter int unsigned HIDDEN_NEURON_MODEL = 0,
    parameter int unsigned OUTPUT_NEURON_MODEL = 0,
    parameter int unsigned HIDDEN_EXECUTOR_COUNT =
        (HIDDEN_NEURON_MODEL == 0) ? HIDDEN_NEURONS : 1,
    parameter int unsigned OUTPUT_EXECUTOR_COUNT =
        (OUTPUT_NEURON_MODEL == 0) ? OUTPUT_NEURONS : 1,
    parameter logic [31:0] INTERFACE_VERSION = 32'h0001_0004,
    parameter logic signed [MEM_WIDTH-1:0] THRESHOLD = 100,
    parameter logic signed [MEM_WIDTH-1:0] RESET_POT = 0,
    parameter int unsigned LEAK_SHIFT = 3,
    parameter int unsigned IZH_FRAC_BITS = 8,
    parameter int unsigned IZH_A_NUMERATOR = 1,
    parameter int unsigned IZH_A_DENOMINATOR = 50,
    parameter int unsigned IZH_B_NUMERATOR = 1,
    parameter int unsigned IZH_B_DENOMINATOR = 5,
    parameter integer IZH_RESET_VOLTAGE = -65,
    parameter integer IZH_RECOVERY_INCREMENT = 8,
    parameter integer IZH_SPIKE_THRESHOLD = 30,
    // Packaged Vivado IP imports the ROMs beside the RTL source and resolves
    // their registered basenames. Standalone simulations use lower-level
    // defaults pointing at data/hh_luts/.
    parameter HH_LUT_DIR = ""
) (
    (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 S_AXI_CLK CLK" *)
    (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME S_AXI_CLK, ASSOCIATED_BUSIF S_AXI, ASSOCIATED_RESET s_axi_aresetn" *)
    input  logic s_axi_aclk,
    (* X_INTERFACE_INFO = "xilinx.com:signal:reset:1.0 S_AXI_RST RST" *)
    (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME S_AXI_RST, POLARITY ACTIVE_LOW" *)
    input  logic s_axi_aresetn,

    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI AWADDR" *)
    (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME S_AXI, PROTOCOL AXI4LITE, READ_WRITE_MODE READ_WRITE" *)
    input  logic [AXI_ADDR_WIDTH-1:0] s_axi_awaddr,
    // verilator lint_off UNUSEDSIGNAL
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI AWPROT" *)
    input  logic [2:0] s_axi_awprot,
    // verilator lint_on UNUSEDSIGNAL
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI AWVALID" *)
    input  logic s_axi_awvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI AWREADY" *)
    output logic s_axi_awready,

    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI WDATA" *)
    input  logic [31:0] s_axi_wdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI WSTRB" *)
    input  logic [3:0] s_axi_wstrb,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI WVALID" *)
    input  logic s_axi_wvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI WREADY" *)
    output logic s_axi_wready,

    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI BRESP" *)
    output logic [1:0] s_axi_bresp,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI BVALID" *)
    output logic s_axi_bvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI BREADY" *)
    input  logic s_axi_bready,

    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI ARADDR" *)
    input  logic [AXI_ADDR_WIDTH-1:0] s_axi_araddr,
    // verilator lint_off UNUSEDSIGNAL
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI ARPROT" *)
    input  logic [2:0] s_axi_arprot,
    // verilator lint_on UNUSEDSIGNAL
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI ARVALID" *)
    input  logic s_axi_arvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI ARREADY" *)
    output logic s_axi_arready,

    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI RDATA" *)
    output logic [31:0] s_axi_rdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI RRESP" *)
    output logic [1:0] s_axi_rresp,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI RVALID" *)
    output logic s_axi_rvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI RREADY" *)
    input  logic s_axi_rready,

    (* X_INTERFACE_INFO = "xilinx.com:signal:interrupt:1.0 IRQ INTERRUPT" *)
    (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME IRQ, SENSITIVITY LEVEL_HIGH" *)
    output logic irq
);
    localparam int unsigned INPUT_WIDTH =
        (INPUT_NEURONS <= 1) ? 1 : $clog2(INPUT_NEURONS);
    localparam int unsigned HIDDEN_WIDTH =
        (HIDDEN_NEURONS <= 1) ? 1 : $clog2(HIDDEN_NEURONS);
    localparam int unsigned OUTPUT_WIDTH =
        (OUTPUT_NEURONS <= 1) ? 1 : $clog2(OUTPUT_NEURONS);
    localparam int unsigned CONFIG_PRE_WIDTH =
        (INPUT_WIDTH > HIDDEN_WIDTH) ? INPUT_WIDTH : HIDDEN_WIDTH;
    localparam int unsigned CONFIG_POST_WIDTH =
        (HIDDEN_WIDTH > OUTPUT_WIDTH) ? HIDDEN_WIDTH : OUTPUT_WIDTH;
    localparam int unsigned EVENT_ADDR_WIDTH =
        (NUM_EVENTS <= 1) ? 1 : $clog2(NUM_EVENTS);
    localparam int unsigned EVENT_COUNT_WIDTH =
        (NUM_EVENTS <= 1) ? 1 : $clog2(NUM_EVENTS + 1);

    localparam logic [31:0] CORE_ID = 32'h534E4E31; // "SNN1"
    localparam logic [31:0] CORE_VERSION = INTERFACE_VERSION;

    localparam logic [AXI_ADDR_WIDTH-1:0] REG_ID = 'h00;
    localparam logic [AXI_ADDR_WIDTH-1:0] REG_VERSION = 'h04;
    localparam logic [AXI_ADDR_WIDTH-1:0] REG_CONTROL = 'h08;
    localparam logic [AXI_ADDR_WIDTH-1:0] REG_STATUS = 'h0C;
    localparam logic [AXI_ADDR_WIDTH-1:0] REG_RESULT = 'h10;
    localparam logic [AXI_ADDR_WIDTH-1:0] REG_WEIGHT_CURSOR = 'h14;
    localparam logic [AXI_ADDR_WIDTH-1:0] REG_WEIGHT_DATA = 'h18;
    localparam logic [AXI_ADDR_WIDTH-1:0] REG_EVENT_CURSOR = 'h1C;
    localparam logic [AXI_ADDR_WIDTH-1:0] REG_EVENT_TIMESTAMP = 'h20;
    localparam logic [AXI_ADDR_WIDTH-1:0] REG_EVENT_NEURON = 'h24;
    localparam logic [AXI_ADDR_WIDTH-1:0] REG_DIMENSIONS_0 = 'h28;
    localparam logic [AXI_ADDR_WIDTH-1:0] REG_DIMENSIONS_1 = 'h2C;
    localparam logic [AXI_ADDR_WIDTH-1:0] REG_NUMERIC_0 = 'h30;
    localparam logic [AXI_ADDR_WIDTH-1:0] REG_NUMERIC_1 = 'h34;
    localparam logic [AXI_ADDR_WIDTH-1:0] REG_THRESHOLD = 'h38;
    localparam logic [AXI_ADDR_WIDTH-1:0] REG_RESET_POT = 'h3C;
    localparam logic [AXI_ADDR_WIDTH-1:0] REG_WEIGHT_CRC = 'h40;
    localparam logic [AXI_ADDR_WIDTH-1:0] REG_EVENT_CRC = 'h44;
    localparam logic [AXI_ADDR_WIDTH-1:0] REG_WEIGHT_COUNT = 'h48;
    localparam logic [AXI_ADDR_WIDTH-1:0] REG_EVENT_COUNT = 'h4C;
    localparam logic [AXI_ADDR_WIDTH-1:0] REG_ERROR = 'h50;
    localparam logic [AXI_ADDR_WIDTH-1:0] REG_NEURON_MODELS = 'h54;
    localparam logic [AXI_ADDR_WIDTH-1:0] REG_IZH_AB = 'h58;
    localparam logic [AXI_ADDR_WIDTH-1:0] REG_IZH_RESET_VOLTAGE = 'h5C;
    localparam logic [AXI_ADDR_WIDTH-1:0] REG_IZH_RECOVERY_INCREMENT = 'h60;
    localparam logic [AXI_ADDR_WIDTH-1:0] REG_IZH_SPIKE_THRESHOLD = 'h64;
    localparam logic [AXI_ADDR_WIDTH-1:0] REG_HH_Q_FORMAT = 'h68;
    localparam logic [AXI_ADDR_WIDTH-1:0] REG_HH_WIDTHS_STEPS = 'h6C;
    localparam logic [AXI_ADDR_WIDTH-1:0] REG_HH_LUT_RANGE = 'h70;
    localparam logic [AXI_ADDR_WIDTH-1:0] REG_HH_SPIKE_THRESHOLD = 'h74;
    localparam logic [AXI_ADDR_WIDTH-1:0] REG_INFERENCE_EVENT_COUNT = 'h78;
    localparam logic [AXI_ADDR_WIDTH-1:0] REG_TIMESTEP_COUNT = 'h7C;
    localparam logic [AXI_ADDR_WIDTH-1:0] REG_INFERENCE_CYCLES_LO = 'h80;
    localparam logic [AXI_ADDR_WIDTH-1:0] REG_INFERENCE_CYCLES_HI = 'h84;
    localparam logic [AXI_ADDR_WIDTH-1:0] REG_CONSUMED_INPUT_EVENTS = 'h88;
    localparam logic [AXI_ADDR_WIDTH-1:0] REG_EMITTED_HIDDEN_SPIKES = 'h8C;
    localparam logic [AXI_ADDR_WIDTH-1:0] REG_EMITTED_OUTPUT_SPIKES = 'h90;
    localparam logic [AXI_ADDR_WIDTH-1:0] REG_EXECUTOR_COUNTS = 'h94;
    localparam logic [AXI_ADDR_WIDTH-1:0] REG_PHASE_SYNAPSE_LO = 'h98;
    localparam logic [AXI_ADDR_WIDTH-1:0] REG_PHASE_SYNAPSE_HI = 'h9C;
    localparam logic [AXI_ADDR_WIDTH-1:0] REG_PHASE_HIDDEN_UPDATE_LO = 'hA0;
    localparam logic [AXI_ADDR_WIDTH-1:0] REG_PHASE_HIDDEN_UPDATE_HI = 'hA4;
    localparam logic [AXI_ADDR_WIDTH-1:0] REG_PHASE_HIDDEN_ROUTING_LO = 'hA8;
    localparam logic [AXI_ADDR_WIDTH-1:0] REG_PHASE_HIDDEN_ROUTING_HI = 'hAC;
    localparam logic [AXI_ADDR_WIDTH-1:0] REG_PHASE_OUTPUT_UPDATE_LO = 'hB0;
    localparam logic [AXI_ADDR_WIDTH-1:0] REG_PHASE_OUTPUT_UPDATE_HI = 'hB4;
    localparam logic [AXI_ADDR_WIDTH-1:0] REG_PHASE_CONTROL_OVERHEAD_LO = 'hB8;
    localparam logic [AXI_ADDR_WIDTH-1:0] REG_PHASE_CONTROL_OVERHEAD_HI = 'hBC;

    localparam logic [1:0] AXI_OKAY = 2'b00;
    localparam logic [1:0] AXI_SLVERR = 2'b10;
    localparam logic [1:0] AXI_DECERR = 2'b11;

    localparam logic [7:0] ERROR_NONE = 8'h00;
    localparam logic [7:0] ERROR_BAD_REGISTER = 8'h01;
    localparam logic [7:0] ERROR_NOT_READY = 8'h02;
    localparam logic [7:0] ERROR_BAD_INDEX = 8'h03;
    localparam logic [7:0] ERROR_BAD_STROBE = 8'h04;
    localparam logic [7:0] ERROR_BAD_CONFIG = 8'h05;

    localparam logic [14:0] INPUT_LAST = 15'(INPUT_NEURONS - 1);
    localparam logic [14:0] HIDDEN_PRE_LAST = 15'(HIDDEN_NEURONS - 1);
    localparam logic [15:0] HIDDEN_POST_LAST = 16'(HIDDEN_NEURONS - 1);
    localparam logic [15:0] OUTPUT_LAST = 16'(OUTPUT_NEURONS - 1);
    localparam logic [15:0] INPUT_NEURON_LAST = 16'(INPUT_NEURONS - 1);
    localparam logic [31:0] EVENT_LAST = 32'(NUM_EVENTS - 1);

    logic aw_pending;
    logic [AXI_ADDR_WIDTH-1:0] awaddr_reg;
    logic w_pending;
    logic [31:0] wdata_reg;
    logic [3:0] wstrb_reg;

    logic weight_cursor_layer;
    logic [14:0] weight_cursor_pre;
    logic [15:0] weight_cursor_post;
    logic [31:0] event_cursor;
    logic [31:0] event_timestamp_reg;

    logic core_start;
    logic core_weight_valid;
    logic core_weight_layer;
    logic [CONFIG_PRE_WIDTH-1:0] core_weight_pre;
    logic [CONFIG_POST_WIDTH-1:0] core_weight_post;
    logic signed [WEIGHT_WIDTH-1:0] core_weight_data;
    logic core_weight_ready;
    logic core_event_valid;
    logic [EVENT_ADDR_WIDTH-1:0] core_event_addr;
    logic [TIME_WIDTH-1:0] core_event_timestamp;
    logic [INPUT_WIDTH-1:0] core_event_neuron;
    logic core_event_ready;
    logic core_inference_done;
    logic core_tx_valid;
    logic [OUTPUT_WIDTH-1:0] core_tx_data;
    logic [63:0] core_inference_cycles;
    logic [31:0] core_consumed_input_events;
    logic [31:0] core_emitted_hidden_spikes;
    logic [31:0] core_emitted_output_spikes;
    logic [63:0] core_synapse_phase_cycles;
    logic [63:0] core_hidden_update_phase_cycles;
    logic [63:0] core_hidden_routing_phase_cycles;
    logic [63:0] core_output_update_phase_cycles;
    logic [63:0] core_control_overhead_phase_cycles;

    logic done_sticky;
    logic result_valid_sticky;
    logic [OUTPUT_WIDTH-1:0] result_reg;
    logic error_sticky;
    logic [7:0] error_code;
    logic [31:0] weight_crc_state;
    logic [31:0] event_crc_state;
    logic [31:0] weight_write_count;
    logic [31:0] event_write_count;
    logic [31:0] inference_event_count;
    logic [TIME_WIDTH-1:0] timestep_count;

    wire core_idle = core_event_ready;
    wire core_busy = !core_idle;
    wire weight_cursor_valid =
        (!weight_cursor_layer &&
         weight_cursor_pre <= INPUT_LAST &&
         weight_cursor_post <= HIDDEN_POST_LAST) ||
        (weight_cursor_layer &&
         weight_cursor_pre <= HIDDEN_PRE_LAST &&
         weight_cursor_post <= OUTPUT_LAST);

    assign s_axi_awready = !aw_pending && !s_axi_bvalid;
    assign s_axi_wready = !w_pending && !s_axi_bvalid;
    assign s_axi_arready = !s_axi_rvalid;
    assign irq = result_valid_sticky || error_sticky;

    function automatic logic [31:0] merge_wstrb(
        input logic [31:0] previous,
        input logic [31:0] update,
        input logic [3:0] strobes
    );
        logic [31:0] merged;
        merged = previous;
        for (int byte_index = 0; byte_index < 4; byte_index++)
            if (strobes[byte_index])
                merged[byte_index*8 +: 8] = update[byte_index*8 +: 8];
        return merged;
    endfunction

    function automatic logic [31:0] crc32_byte(
        input logic [31:0] crc_in,
        input logic [7:0] data
    );
        logic [31:0] crc;
        crc = crc_in ^ {24'b0, data};
        for (int bit_index = 0; bit_index < 8; bit_index++)
            crc = crc[0] ? ((crc >> 1) ^ 32'hEDB8_8320) : (crc >> 1);
        return crc;
    endfunction

    function automatic logic [31:0] crc32_event(
        input logic [31:0] crc_in,
        input logic [31:0] timestamp,
        input logic [15:0] neuron
    );
        logic [31:0] crc;
        crc = crc32_byte(crc_in, timestamp[7:0]);
        crc = crc32_byte(crc, timestamp[15:8]);
        crc = crc32_byte(crc, timestamp[23:16]);
        crc = crc32_byte(crc, timestamp[31:24]);
        crc = crc32_byte(crc, neuron[7:0]);
        crc = crc32_byte(crc, neuron[15:8]);
        return crc;
    endfunction

    initial begin
        if (AXI_ADDR_WIDTH < 8)
            $error("AXI_ADDR_WIDTH must address registers through 0x94");
        if (INPUT_NEURONS > 32768 || HIDDEN_NEURONS > 32768 ||
            OUTPUT_NEURONS > 65536)
            $error("network dimensions exceed cursor fields");
        if (WEIGHT_WIDTH > 8)
            $error("AXI wrapper currently supports weights up to 8 bits");
        if (TIME_WIDTH > 32 || INPUT_WIDTH > 16 || OUTPUT_WIDTH > 32)
            $error("event register fields are too narrow for this configuration");
    end

    always_ff @(posedge s_axi_aclk) begin
        if (!s_axi_aresetn) begin
            aw_pending <= 1'b0;
            awaddr_reg <= '0;
            w_pending <= 1'b0;
            wdata_reg <= '0;
            wstrb_reg <= '0;
            s_axi_bresp <= AXI_OKAY;
            s_axi_bvalid <= 1'b0;
            s_axi_rdata <= '0;
            s_axi_rresp <= AXI_OKAY;
            s_axi_rvalid <= 1'b0;

            weight_cursor_layer <= 1'b0;
            weight_cursor_pre <= '0;
            weight_cursor_post <= '0;
            event_cursor <= '0;
            event_timestamp_reg <= '0;

            core_start <= 1'b0;
            core_weight_valid <= 1'b0;
            core_weight_layer <= 1'b0;
            core_weight_pre <= '0;
            core_weight_post <= '0;
            core_weight_data <= '0;
            core_event_valid <= 1'b0;
            core_event_addr <= '0;
            core_event_timestamp <= '0;
            core_event_neuron <= '0;

            done_sticky <= 1'b0;
            result_valid_sticky <= 1'b0;
            result_reg <= '0;
            error_sticky <= 1'b0;
            error_code <= ERROR_NONE;
            weight_crc_state <= 32'hFFFF_FFFF;
            event_crc_state <= 32'hFFFF_FFFF;
            weight_write_count <= '0;
            event_write_count <= '0;
            inference_event_count <= 32'(NUM_EVENTS);
            timestep_count <= TIME_WIDTH'(1);
        end
        else begin
            core_start <= 1'b0;
            core_weight_valid <= 1'b0;
            core_event_valid <= 1'b0;

            if (s_axi_awready && s_axi_awvalid) begin
                aw_pending <= 1'b1;
                awaddr_reg <= s_axi_awaddr;
            end
            if (s_axi_wready && s_axi_wvalid) begin
                w_pending <= 1'b1;
                wdata_reg <= s_axi_wdata;
                wstrb_reg <= s_axi_wstrb;
            end
            if (s_axi_bvalid && s_axi_bready)
                s_axi_bvalid <= 1'b0;

            if (core_inference_done)
                done_sticky <= 1'b1;
            if (core_tx_valid) begin
                result_reg <= core_tx_data;
                result_valid_sticky <= 1'b1;
            end

            if (aw_pending && w_pending && !s_axi_bvalid) begin
                aw_pending <= 1'b0;
                w_pending <= 1'b0;
                s_axi_bvalid <= 1'b1;
                s_axi_bresp <= AXI_OKAY;

                case (awaddr_reg)
                    REG_CONTROL: begin
                        if (!wstrb_reg[0]) begin
                            s_axi_bresp <= AXI_SLVERR;
                            error_sticky <= 1'b1;
                            error_code <= ERROR_BAD_STROBE;
                        end
                        else begin
                            if (wdata_reg[1]) begin
                                done_sticky <= 1'b0;
                                result_valid_sticky <= 1'b0;
                                error_sticky <= 1'b0;
                                error_code <= ERROR_NONE;
                                weight_crc_state <= 32'hFFFF_FFFF;
                                event_crc_state <= 32'hFFFF_FFFF;
                                weight_write_count <= '0;
                                event_write_count <= '0;
                            end
                            if (wdata_reg[0]) begin
                                if (!core_idle) begin
                                    s_axi_bresp <= AXI_SLVERR;
                                    error_sticky <= 1'b1;
                                    error_code <= ERROR_NOT_READY;
                                end
                                else if (inference_event_count > NUM_EVENTS ||
                                         timestep_count == '0 ||
                                         event_write_count != inference_event_count) begin
                                    s_axi_bresp <= AXI_SLVERR;
                                    error_sticky <= 1'b1;
                                    error_code <= ERROR_BAD_CONFIG;
                                end
                                else begin
                                    core_start <= 1'b1;
                                    // A successful START acknowledges the old
                                    // completion/result snapshot while keeping
                                    // event memory and upload checks intact.
                                    // This permits measurement replays without
                                    // another event upload.
                                    done_sticky <= 1'b0;
                                    result_valid_sticky <= 1'b0;
                                end
                            end
                        end
                    end

                    REG_WEIGHT_CURSOR: begin
                        logic [31:0] merged;
                        merged = merge_wstrb(
                            {weight_cursor_layer, weight_cursor_pre, weight_cursor_post},
                            wdata_reg,
                            wstrb_reg
                        );
                        weight_cursor_layer <= merged[31];
                        weight_cursor_pre <= merged[30:16];
                        weight_cursor_post <= merged[15:0];
                    end

                    REG_WEIGHT_DATA: begin
                        if (!wstrb_reg[0]) begin
                            s_axi_bresp <= AXI_SLVERR;
                            error_sticky <= 1'b1;
                            error_code <= ERROR_BAD_STROBE;
                        end
                        else if (!weight_cursor_valid) begin
                            s_axi_bresp <= AXI_SLVERR;
                            error_sticky <= 1'b1;
                            error_code <= ERROR_BAD_INDEX;
                        end
                        else if (!core_weight_ready) begin
                            s_axi_bresp <= AXI_SLVERR;
                            error_sticky <= 1'b1;
                            error_code <= ERROR_NOT_READY;
                        end
                        else begin
                            core_weight_layer <= weight_cursor_layer;
                            core_weight_pre <=
                                weight_cursor_pre[CONFIG_PRE_WIDTH-1:0];
                            core_weight_post <=
                                weight_cursor_post[CONFIG_POST_WIDTH-1:0];
                            core_weight_data <= wdata_reg[WEIGHT_WIDTH-1:0];
                            core_weight_valid <= 1'b1;
                            weight_crc_state <= crc32_byte(weight_crc_state, wdata_reg[7:0]);
                            weight_write_count <= weight_write_count + 1'b1;

                            if (!weight_cursor_layer) begin
                                if (weight_cursor_post == HIDDEN_POST_LAST) begin
                                    weight_cursor_post <= '0;
                                    weight_cursor_pre <=
                                        (weight_cursor_pre == INPUT_LAST) ? '0 :
                                        weight_cursor_pre + 1'b1;
                                end
                                else
                                    weight_cursor_post <= weight_cursor_post + 1'b1;
                            end
                            else begin
                                if (weight_cursor_post == OUTPUT_LAST) begin
                                    weight_cursor_post <= '0;
                                    weight_cursor_pre <=
                                        (weight_cursor_pre == HIDDEN_PRE_LAST) ? '0 :
                                        weight_cursor_pre + 1'b1;
                                end
                                else
                                    weight_cursor_post <= weight_cursor_post + 1'b1;
                            end
                        end
                    end

                    REG_EVENT_CURSOR:
                        event_cursor <= merge_wstrb(event_cursor, wdata_reg, wstrb_reg);

                    REG_EVENT_TIMESTAMP:
                        event_timestamp_reg <=
                            merge_wstrb(event_timestamp_reg, wdata_reg, wstrb_reg);

                    REG_EVENT_NEURON: begin
                        if (wstrb_reg[1:0] != 2'b11) begin
                            s_axi_bresp <= AXI_SLVERR;
                            error_sticky <= 1'b1;
                            error_code <= ERROR_BAD_STROBE;
                        end
                        else if (event_cursor >= NUM_EVENTS ||
                                 wdata_reg[15:0] > INPUT_NEURON_LAST) begin
                            s_axi_bresp <= AXI_SLVERR;
                            error_sticky <= 1'b1;
                            error_code <= ERROR_BAD_INDEX;
                        end
                        else if (!core_event_ready) begin
                            s_axi_bresp <= AXI_SLVERR;
                            error_sticky <= 1'b1;
                            error_code <= ERROR_NOT_READY;
                        end
                        else begin
                            core_event_addr <= event_cursor[EVENT_ADDR_WIDTH-1:0];
                            core_event_timestamp <= event_timestamp_reg[TIME_WIDTH-1:0];
                            core_event_neuron <= wdata_reg[INPUT_WIDTH-1:0];
                            core_event_valid <= 1'b1;
                            event_crc_state <= crc32_event(
                                event_crc_state,
                                event_timestamp_reg,
                                wdata_reg[15:0]
                            );
                            event_write_count <= event_write_count + 1'b1;
                            event_cursor <= (event_cursor == EVENT_LAST) ? '0 :
                                            event_cursor + 1'b1;
                        end
                    end

                    REG_INFERENCE_EVENT_COUNT: begin
                        if (!core_idle) begin
                            s_axi_bresp <= AXI_SLVERR;
                            error_sticky <= 1'b1;
                            error_code <= ERROR_NOT_READY;
                        end
                        else
                            inference_event_count <= merge_wstrb(
                                inference_event_count, wdata_reg, wstrb_reg
                            );
                    end

                    REG_TIMESTEP_COUNT: begin
                        logic [31:0] merged;
                        if (!core_idle) begin
                            s_axi_bresp <= AXI_SLVERR;
                            error_sticky <= 1'b1;
                            error_code <= ERROR_NOT_READY;
                        end
                        else begin
                            merged = merge_wstrb(
                                32'(timestep_count), wdata_reg, wstrb_reg
                            );
                            timestep_count <= merged[TIME_WIDTH-1:0];
                        end
                    end

                    default: begin
                        s_axi_bresp <= AXI_DECERR;
                        error_sticky <= 1'b1;
                        error_code <= ERROR_BAD_REGISTER;
                    end
                endcase
            end

            if (s_axi_rvalid && s_axi_rready)
                s_axi_rvalid <= 1'b0;
            if (s_axi_arready && s_axi_arvalid) begin
                s_axi_rvalid <= 1'b1;
                s_axi_rresp <= AXI_OKAY;
                case (s_axi_araddr)
                    REG_ID: s_axi_rdata <= CORE_ID;
                    REG_VERSION: s_axi_rdata <= CORE_VERSION;
                    REG_CONTROL: s_axi_rdata <= '0;
                    REG_STATUS: begin
                        s_axi_rdata <= '0;
                        s_axi_rdata[0] <= core_idle;
                        s_axi_rdata[1] <= core_busy;
                        s_axi_rdata[2] <= done_sticky;
                        s_axi_rdata[3] <= result_valid_sticky;
                        s_axi_rdata[4] <= error_sticky;
                        s_axi_rdata[5] <= core_weight_ready;
                        s_axi_rdata[6] <= core_event_ready;
                    end
                    REG_RESULT: s_axi_rdata <= {{(32-OUTPUT_WIDTH){1'b0}}, result_reg};
                    REG_WEIGHT_CURSOR:
                        s_axi_rdata <= {weight_cursor_layer, weight_cursor_pre,
                                       weight_cursor_post};
                    REG_WEIGHT_DATA: s_axi_rdata <= '0;
                    REG_EVENT_CURSOR: s_axi_rdata <= event_cursor;
                    REG_EVENT_TIMESTAMP: s_axi_rdata <= event_timestamp_reg;
                    REG_EVENT_NEURON: s_axi_rdata <= '0;
                    REG_DIMENSIONS_0:
                        s_axi_rdata <= {16'(HIDDEN_NEURONS), 16'(INPUT_NEURONS)};
                    REG_DIMENSIONS_1:
                        s_axi_rdata <= {16'(NUM_EVENTS), 16'(OUTPUT_NEURONS)};
                    REG_NUMERIC_0:
                        s_axi_rdata <= {8'(COUNT_WIDTH), 8'(CURRENT_WIDTH),
                                       8'(MEM_WIDTH), 8'(WEIGHT_WIDTH)};
                    REG_NUMERIC_1:
                        s_axi_rdata <= {16'b0, 8'(LEAK_SHIFT), 8'(TIME_WIDTH)};
                    REG_THRESHOLD: s_axi_rdata <= 32'(THRESHOLD);
                    REG_RESET_POT: s_axi_rdata <= 32'(RESET_POT);
                    REG_WEIGHT_CRC: s_axi_rdata <= weight_crc_state ^ 32'hFFFF_FFFF;
                    REG_EVENT_CRC: s_axi_rdata <= event_crc_state ^ 32'hFFFF_FFFF;
                    REG_WEIGHT_COUNT: s_axi_rdata <= weight_write_count;
                    REG_EVENT_COUNT: s_axi_rdata <= event_write_count;
                    REG_ERROR: s_axi_rdata <= {24'b0, error_code};
                    REG_NEURON_MODELS:
                        s_axi_rdata <= {8'(IZH_FRAC_BITS), 8'b0,
                                       8'(OUTPUT_NEURON_MODEL),
                                       8'(HIDDEN_NEURON_MODEL)};
                    REG_IZH_AB:
                        s_axi_rdata <= {8'(IZH_B_DENOMINATOR),
                                       8'(IZH_B_NUMERATOR),
                                       8'(IZH_A_DENOMINATOR),
                                       8'(IZH_A_NUMERATOR)};
                    REG_IZH_RESET_VOLTAGE:
                        s_axi_rdata <= 32'(IZH_RESET_VOLTAGE);
                    REG_IZH_RECOVERY_INCREMENT:
                        s_axi_rdata <= 32'(IZH_RECOVERY_INCREMENT);
                    REG_IZH_SPIKE_THRESHOLD:
                        s_axi_rdata <= 32'(IZH_SPIKE_THRESHOLD);
                    REG_HH_Q_FORMAT:
                        s_axi_rdata <= {8'd2, 8'd18, 8'd18, 8'd12};
                    REG_HH_WIDTHS_STEPS:
                        s_axi_rdata <= {8'd100, 8'd24, 8'd20, 8'd24};
                    REG_HH_LUT_RANGE:
                        s_axi_rdata <= {16'sd128, -16'sd32};
                    REG_HH_SPIKE_THRESHOLD:
                        s_axi_rdata <= 32'sd20;
                    REG_INFERENCE_EVENT_COUNT:
                        s_axi_rdata <= inference_event_count;
                    REG_TIMESTEP_COUNT:
                        s_axi_rdata <= 32'(timestep_count);
                    REG_INFERENCE_CYCLES_LO:
                        s_axi_rdata <= core_inference_cycles[31:0];
                    REG_INFERENCE_CYCLES_HI:
                        s_axi_rdata <= core_inference_cycles[63:32];
                    REG_CONSUMED_INPUT_EVENTS:
                        s_axi_rdata <= core_consumed_input_events;
                    REG_EMITTED_HIDDEN_SPIKES:
                        s_axi_rdata <= core_emitted_hidden_spikes;
                    REG_EMITTED_OUTPUT_SPIKES:
                        s_axi_rdata <= core_emitted_output_spikes;
                    REG_EXECUTOR_COUNTS:
                        s_axi_rdata <= {16'(HIDDEN_EXECUTOR_COUNT),
                                       16'(OUTPUT_EXECUTOR_COUNT)};
                    REG_PHASE_SYNAPSE_LO:
                        s_axi_rdata <= core_synapse_phase_cycles[31:0];
                    REG_PHASE_SYNAPSE_HI:
                        s_axi_rdata <= core_synapse_phase_cycles[63:32];
                    REG_PHASE_HIDDEN_UPDATE_LO:
                        s_axi_rdata <= core_hidden_update_phase_cycles[31:0];
                    REG_PHASE_HIDDEN_UPDATE_HI:
                        s_axi_rdata <= core_hidden_update_phase_cycles[63:32];
                    REG_PHASE_HIDDEN_ROUTING_LO:
                        s_axi_rdata <= core_hidden_routing_phase_cycles[31:0];
                    REG_PHASE_HIDDEN_ROUTING_HI:
                        s_axi_rdata <= core_hidden_routing_phase_cycles[63:32];
                    REG_PHASE_OUTPUT_UPDATE_LO:
                        s_axi_rdata <= core_output_update_phase_cycles[31:0];
                    REG_PHASE_OUTPUT_UPDATE_HI:
                        s_axi_rdata <= core_output_update_phase_cycles[63:32];
                    REG_PHASE_CONTROL_OVERHEAD_LO:
                        s_axi_rdata <= core_control_overhead_phase_cycles[31:0];
                    REG_PHASE_CONTROL_OVERHEAD_HI:
                        s_axi_rdata <= core_control_overhead_phase_cycles[63:32];
                    default: begin
                        s_axi_rdata <= '0;
                        s_axi_rresp <= AXI_DECERR;
                    end
                endcase
            end
        end
    end

    snn_top #(
        .NUM_EVENTS(NUM_EVENTS),
        .INPUT_NEURONS(INPUT_NEURONS),
        .HIDDEN_NEURONS(HIDDEN_NEURONS),
        .OUTPUT_NEURONS(OUTPUT_NEURONS),
        .INPUT_WIDTH(INPUT_WIDTH),
        .HIDDEN_WIDTH(HIDDEN_WIDTH),
        .OUTPUT_WIDTH(OUTPUT_WIDTH),
        .CONFIG_PRE_WIDTH(CONFIG_PRE_WIDTH),
        .CONFIG_POST_WIDTH(CONFIG_POST_WIDTH),
        .TIME_WIDTH(TIME_WIDTH),
        .MEM_WIDTH(MEM_WIDTH),
        .CURRENT_WIDTH(CURRENT_WIDTH),
        .WEIGHT_WIDTH(WEIGHT_WIDTH),
        .COUNT_WIDTH(COUNT_WIDTH),
        .HIDDEN_NEURON_MODEL(HIDDEN_NEURON_MODEL),
        .OUTPUT_NEURON_MODEL(OUTPUT_NEURON_MODEL),
        .HIDDEN_EXECUTOR_COUNT(HIDDEN_EXECUTOR_COUNT),
        .OUTPUT_EXECUTOR_COUNT(OUTPUT_EXECUTOR_COUNT),
        .THRESHOLD(THRESHOLD),
        .RESET_POT(RESET_POT),
        .LEAK_SHIFT(LEAK_SHIFT),
        .IZH_FRAC_BITS(IZH_FRAC_BITS),
        .IZH_A_NUMERATOR(IZH_A_NUMERATOR),
        .IZH_A_DENOMINATOR(IZH_A_DENOMINATOR),
        .IZH_B_NUMERATOR(IZH_B_NUMERATOR),
        .IZH_B_DENOMINATOR(IZH_B_DENOMINATOR),
        .IZH_RESET_VOLTAGE(IZH_RESET_VOLTAGE),
        .IZH_RECOVERY_INCREMENT(IZH_RECOVERY_INCREMENT),
        .IZH_SPIKE_THRESHOLD(IZH_SPIKE_THRESHOLD),
        .HH_LUT_DIR(HH_LUT_DIR)
    ) u_snn_top (
        .clk(s_axi_aclk),
        .rst(!s_axi_aresetn),
        .start_inference(core_start),
        .input_event_count(inference_event_count[EVENT_COUNT_WIDTH-1:0]),
        .timestep_count(timestep_count),
        .weight_write_valid(core_weight_valid),
        .weight_layer(core_weight_layer),
        .weight_pre_idx(core_weight_pre),
        .weight_post_idx(core_weight_post),
        .weight_write_data(core_weight_data),
        .weight_write_ready(core_weight_ready),
        .spike_write_valid(core_event_valid),
        .spike_write_addr(core_event_addr),
        .spike_write_timestamp(core_event_timestamp),
        .spike_write_neuron_id(core_event_neuron),
        .spike_write_ready(core_event_ready),
        .inference_done(core_inference_done),
        .tx_valid(core_tx_valid),
        .tx_data(core_tx_data),
        .inference_cycles(core_inference_cycles),
        .consumed_input_events(core_consumed_input_events),
        .emitted_hidden_spikes(core_emitted_hidden_spikes),
        .emitted_output_spikes(core_emitted_output_spikes),
        .synapse_phase_cycles(core_synapse_phase_cycles),
        .hidden_update_phase_cycles(core_hidden_update_phase_cycles),
        .hidden_routing_phase_cycles(core_hidden_routing_phase_cycles),
        .output_update_phase_cycles(core_output_update_phase_cycles),
        .control_overhead_phase_cycles(core_control_overhead_phase_cycles)
    );
endmodule
