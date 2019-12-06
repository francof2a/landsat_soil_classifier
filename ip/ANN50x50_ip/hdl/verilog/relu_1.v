// ==============================================================
// RTL generated by Vivado(TM) HLS - High-Level Synthesis from C, C++ and SystemC
// Version: 2019.1
// Copyright (C) 1986-2019 Xilinx, Inc. All Rights Reserved.
// 
// ===========================================================

`timescale 1 ns / 1 ps 

module relu_1 (
        ap_ready,
        data_0_V_read,
        data_1_V_read,
        data_2_V_read,
        data_3_V_read,
        data_4_V_read,
        data_5_V_read,
        data_6_V_read,
        data_7_V_read,
        data_8_V_read,
        data_9_V_read,
        data_10_V_read,
        data_11_V_read,
        data_12_V_read,
        data_13_V_read,
        data_14_V_read,
        data_15_V_read,
        data_16_V_read,
        data_17_V_read,
        data_18_V_read,
        data_19_V_read,
        data_20_V_read,
        data_21_V_read,
        data_22_V_read,
        data_23_V_read,
        data_24_V_read,
        data_25_V_read,
        data_26_V_read,
        data_27_V_read,
        data_28_V_read,
        data_29_V_read,
        data_30_V_read,
        data_31_V_read,
        data_32_V_read,
        data_33_V_read,
        data_34_V_read,
        data_35_V_read,
        data_36_V_read,
        data_37_V_read,
        data_38_V_read,
        data_39_V_read,
        data_40_V_read,
        data_41_V_read,
        data_42_V_read,
        data_43_V_read,
        data_44_V_read,
        data_45_V_read,
        data_46_V_read,
        data_47_V_read,
        data_48_V_read,
        data_49_V_read,
        ap_return_0,
        ap_return_1,
        ap_return_2,
        ap_return_3,
        ap_return_4,
        ap_return_5,
        ap_return_6,
        ap_return_7,
        ap_return_8,
        ap_return_9,
        ap_return_10,
        ap_return_11,
        ap_return_12,
        ap_return_13,
        ap_return_14,
        ap_return_15,
        ap_return_16,
        ap_return_17,
        ap_return_18,
        ap_return_19,
        ap_return_20,
        ap_return_21,
        ap_return_22,
        ap_return_23,
        ap_return_24,
        ap_return_25,
        ap_return_26,
        ap_return_27,
        ap_return_28,
        ap_return_29,
        ap_return_30,
        ap_return_31,
        ap_return_32,
        ap_return_33,
        ap_return_34,
        ap_return_35,
        ap_return_36,
        ap_return_37,
        ap_return_38,
        ap_return_39,
        ap_return_40,
        ap_return_41,
        ap_return_42,
        ap_return_43,
        ap_return_44,
        ap_return_45,
        ap_return_46,
        ap_return_47,
        ap_return_48,
        ap_return_49
);


output   ap_ready;
input  [31:0] data_0_V_read;
input  [31:0] data_1_V_read;
input  [31:0] data_2_V_read;
input  [31:0] data_3_V_read;
input  [31:0] data_4_V_read;
input  [31:0] data_5_V_read;
input  [31:0] data_6_V_read;
input  [31:0] data_7_V_read;
input  [31:0] data_8_V_read;
input  [31:0] data_9_V_read;
input  [31:0] data_10_V_read;
input  [31:0] data_11_V_read;
input  [31:0] data_12_V_read;
input  [31:0] data_13_V_read;
input  [31:0] data_14_V_read;
input  [31:0] data_15_V_read;
input  [31:0] data_16_V_read;
input  [31:0] data_17_V_read;
input  [31:0] data_18_V_read;
input  [31:0] data_19_V_read;
input  [31:0] data_20_V_read;
input  [31:0] data_21_V_read;
input  [31:0] data_22_V_read;
input  [31:0] data_23_V_read;
input  [31:0] data_24_V_read;
input  [31:0] data_25_V_read;
input  [31:0] data_26_V_read;
input  [31:0] data_27_V_read;
input  [31:0] data_28_V_read;
input  [31:0] data_29_V_read;
input  [31:0] data_30_V_read;
input  [31:0] data_31_V_read;
input  [31:0] data_32_V_read;
input  [31:0] data_33_V_read;
input  [31:0] data_34_V_read;
input  [31:0] data_35_V_read;
input  [31:0] data_36_V_read;
input  [31:0] data_37_V_read;
input  [31:0] data_38_V_read;
input  [31:0] data_39_V_read;
input  [31:0] data_40_V_read;
input  [31:0] data_41_V_read;
input  [31:0] data_42_V_read;
input  [31:0] data_43_V_read;
input  [31:0] data_44_V_read;
input  [31:0] data_45_V_read;
input  [31:0] data_46_V_read;
input  [31:0] data_47_V_read;
input  [31:0] data_48_V_read;
input  [31:0] data_49_V_read;
output  [31:0] ap_return_0;
output  [31:0] ap_return_1;
output  [31:0] ap_return_2;
output  [31:0] ap_return_3;
output  [31:0] ap_return_4;
output  [31:0] ap_return_5;
output  [31:0] ap_return_6;
output  [31:0] ap_return_7;
output  [31:0] ap_return_8;
output  [31:0] ap_return_9;
output  [31:0] ap_return_10;
output  [31:0] ap_return_11;
output  [31:0] ap_return_12;
output  [31:0] ap_return_13;
output  [31:0] ap_return_14;
output  [31:0] ap_return_15;
output  [31:0] ap_return_16;
output  [31:0] ap_return_17;
output  [31:0] ap_return_18;
output  [31:0] ap_return_19;
output  [31:0] ap_return_20;
output  [31:0] ap_return_21;
output  [31:0] ap_return_22;
output  [31:0] ap_return_23;
output  [31:0] ap_return_24;
output  [31:0] ap_return_25;
output  [31:0] ap_return_26;
output  [31:0] ap_return_27;
output  [31:0] ap_return_28;
output  [31:0] ap_return_29;
output  [31:0] ap_return_30;
output  [31:0] ap_return_31;
output  [31:0] ap_return_32;
output  [31:0] ap_return_33;
output  [31:0] ap_return_34;
output  [31:0] ap_return_35;
output  [31:0] ap_return_36;
output  [31:0] ap_return_37;
output  [31:0] ap_return_38;
output  [31:0] ap_return_39;
output  [31:0] ap_return_40;
output  [31:0] ap_return_41;
output  [31:0] ap_return_42;
output  [31:0] ap_return_43;
output  [31:0] ap_return_44;
output  [31:0] ap_return_45;
output  [31:0] ap_return_46;
output  [31:0] ap_return_47;
output  [31:0] ap_return_48;
output  [31:0] ap_return_49;

wire   [0:0] icmp_ln1494_fu_422_p2;
wire   [30:0] trunc_ln83_fu_428_p1;
wire   [30:0] select_ln83_fu_432_p3;
wire   [0:0] icmp_ln1494_1_fu_444_p2;
wire   [30:0] trunc_ln83_1_fu_450_p1;
wire   [30:0] select_ln83_1_fu_454_p3;
wire   [0:0] icmp_ln1494_2_fu_466_p2;
wire   [30:0] trunc_ln83_2_fu_472_p1;
wire   [30:0] select_ln83_2_fu_476_p3;
wire   [0:0] icmp_ln1494_3_fu_488_p2;
wire   [30:0] trunc_ln83_3_fu_494_p1;
wire   [30:0] select_ln83_3_fu_498_p3;
wire   [0:0] icmp_ln1494_4_fu_510_p2;
wire   [30:0] trunc_ln83_4_fu_516_p1;
wire   [30:0] select_ln83_4_fu_520_p3;
wire   [0:0] icmp_ln1494_5_fu_532_p2;
wire   [30:0] trunc_ln83_5_fu_538_p1;
wire   [30:0] select_ln83_5_fu_542_p3;
wire   [0:0] icmp_ln1494_6_fu_554_p2;
wire   [30:0] trunc_ln83_6_fu_560_p1;
wire   [30:0] select_ln83_6_fu_564_p3;
wire   [0:0] icmp_ln1494_7_fu_576_p2;
wire   [30:0] trunc_ln83_7_fu_582_p1;
wire   [30:0] select_ln83_7_fu_586_p3;
wire   [0:0] icmp_ln1494_8_fu_598_p2;
wire   [30:0] trunc_ln83_8_fu_604_p1;
wire   [30:0] select_ln83_8_fu_608_p3;
wire   [0:0] icmp_ln1494_9_fu_620_p2;
wire   [30:0] trunc_ln83_9_fu_626_p1;
wire   [30:0] select_ln83_9_fu_630_p3;
wire   [0:0] icmp_ln1494_10_fu_642_p2;
wire   [30:0] trunc_ln83_10_fu_648_p1;
wire   [30:0] select_ln83_10_fu_652_p3;
wire   [0:0] icmp_ln1494_11_fu_664_p2;
wire   [30:0] trunc_ln83_11_fu_670_p1;
wire   [30:0] select_ln83_11_fu_674_p3;
wire   [0:0] icmp_ln1494_12_fu_686_p2;
wire   [30:0] trunc_ln83_12_fu_692_p1;
wire   [30:0] select_ln83_12_fu_696_p3;
wire   [0:0] icmp_ln1494_13_fu_708_p2;
wire   [30:0] trunc_ln83_13_fu_714_p1;
wire   [30:0] select_ln83_13_fu_718_p3;
wire   [0:0] icmp_ln1494_14_fu_730_p2;
wire   [30:0] trunc_ln83_14_fu_736_p1;
wire   [30:0] select_ln83_14_fu_740_p3;
wire   [0:0] icmp_ln1494_15_fu_752_p2;
wire   [30:0] trunc_ln83_15_fu_758_p1;
wire   [30:0] select_ln83_15_fu_762_p3;
wire   [0:0] icmp_ln1494_16_fu_774_p2;
wire   [30:0] trunc_ln83_16_fu_780_p1;
wire   [30:0] select_ln83_16_fu_784_p3;
wire   [0:0] icmp_ln1494_17_fu_796_p2;
wire   [30:0] trunc_ln83_17_fu_802_p1;
wire   [30:0] select_ln83_17_fu_806_p3;
wire   [0:0] icmp_ln1494_18_fu_818_p2;
wire   [30:0] trunc_ln83_18_fu_824_p1;
wire   [30:0] select_ln83_18_fu_828_p3;
wire   [0:0] icmp_ln1494_19_fu_840_p2;
wire   [30:0] trunc_ln83_19_fu_846_p1;
wire   [30:0] select_ln83_19_fu_850_p3;
wire   [0:0] icmp_ln1494_20_fu_862_p2;
wire   [30:0] trunc_ln83_20_fu_868_p1;
wire   [30:0] select_ln83_20_fu_872_p3;
wire   [0:0] icmp_ln1494_21_fu_884_p2;
wire   [30:0] trunc_ln83_21_fu_890_p1;
wire   [30:0] select_ln83_21_fu_894_p3;
wire   [0:0] icmp_ln1494_22_fu_906_p2;
wire   [30:0] trunc_ln83_22_fu_912_p1;
wire   [30:0] select_ln83_22_fu_916_p3;
wire   [0:0] icmp_ln1494_23_fu_928_p2;
wire   [30:0] trunc_ln83_23_fu_934_p1;
wire   [30:0] select_ln83_23_fu_938_p3;
wire   [0:0] icmp_ln1494_24_fu_950_p2;
wire   [30:0] trunc_ln83_24_fu_956_p1;
wire   [30:0] select_ln83_24_fu_960_p3;
wire   [0:0] icmp_ln1494_25_fu_972_p2;
wire   [30:0] trunc_ln83_25_fu_978_p1;
wire   [30:0] select_ln83_25_fu_982_p3;
wire   [0:0] icmp_ln1494_26_fu_994_p2;
wire   [30:0] trunc_ln83_26_fu_1000_p1;
wire   [30:0] select_ln83_26_fu_1004_p3;
wire   [0:0] icmp_ln1494_27_fu_1016_p2;
wire   [30:0] trunc_ln83_27_fu_1022_p1;
wire   [30:0] select_ln83_27_fu_1026_p3;
wire   [0:0] icmp_ln1494_28_fu_1038_p2;
wire   [30:0] trunc_ln83_28_fu_1044_p1;
wire   [30:0] select_ln83_28_fu_1048_p3;
wire   [0:0] icmp_ln1494_29_fu_1060_p2;
wire   [30:0] trunc_ln83_29_fu_1066_p1;
wire   [30:0] select_ln83_29_fu_1070_p3;
wire   [0:0] icmp_ln1494_30_fu_1082_p2;
wire   [30:0] trunc_ln83_30_fu_1088_p1;
wire   [30:0] select_ln83_30_fu_1092_p3;
wire   [0:0] icmp_ln1494_31_fu_1104_p2;
wire   [30:0] trunc_ln83_31_fu_1110_p1;
wire   [30:0] select_ln83_31_fu_1114_p3;
wire   [0:0] icmp_ln1494_32_fu_1126_p2;
wire   [30:0] trunc_ln83_32_fu_1132_p1;
wire   [30:0] select_ln83_32_fu_1136_p3;
wire   [0:0] icmp_ln1494_33_fu_1148_p2;
wire   [30:0] trunc_ln83_33_fu_1154_p1;
wire   [30:0] select_ln83_33_fu_1158_p3;
wire   [0:0] icmp_ln1494_34_fu_1170_p2;
wire   [30:0] trunc_ln83_34_fu_1176_p1;
wire   [30:0] select_ln83_34_fu_1180_p3;
wire   [0:0] icmp_ln1494_35_fu_1192_p2;
wire   [30:0] trunc_ln83_35_fu_1198_p1;
wire   [30:0] select_ln83_35_fu_1202_p3;
wire   [0:0] icmp_ln1494_36_fu_1214_p2;
wire   [30:0] trunc_ln83_36_fu_1220_p1;
wire   [30:0] select_ln83_36_fu_1224_p3;
wire   [0:0] icmp_ln1494_37_fu_1236_p2;
wire   [30:0] trunc_ln83_37_fu_1242_p1;
wire   [30:0] select_ln83_37_fu_1246_p3;
wire   [0:0] icmp_ln1494_38_fu_1258_p2;
wire   [30:0] trunc_ln83_38_fu_1264_p1;
wire   [30:0] select_ln83_38_fu_1268_p3;
wire   [0:0] icmp_ln1494_39_fu_1280_p2;
wire   [30:0] trunc_ln83_39_fu_1286_p1;
wire   [30:0] select_ln83_39_fu_1290_p3;
wire   [0:0] icmp_ln1494_40_fu_1302_p2;
wire   [30:0] trunc_ln83_40_fu_1308_p1;
wire   [30:0] select_ln83_40_fu_1312_p3;
wire   [0:0] icmp_ln1494_41_fu_1324_p2;
wire   [30:0] trunc_ln83_41_fu_1330_p1;
wire   [30:0] select_ln83_41_fu_1334_p3;
wire   [0:0] icmp_ln1494_42_fu_1346_p2;
wire   [30:0] trunc_ln83_42_fu_1352_p1;
wire   [30:0] select_ln83_42_fu_1356_p3;
wire   [0:0] icmp_ln1494_43_fu_1368_p2;
wire   [30:0] trunc_ln83_43_fu_1374_p1;
wire   [30:0] select_ln83_43_fu_1378_p3;
wire   [0:0] icmp_ln1494_44_fu_1390_p2;
wire   [30:0] trunc_ln83_44_fu_1396_p1;
wire   [30:0] select_ln83_44_fu_1400_p3;
wire   [0:0] icmp_ln1494_45_fu_1412_p2;
wire   [30:0] trunc_ln83_45_fu_1418_p1;
wire   [30:0] select_ln83_45_fu_1422_p3;
wire   [0:0] icmp_ln1494_46_fu_1434_p2;
wire   [30:0] trunc_ln83_46_fu_1440_p1;
wire   [30:0] select_ln83_46_fu_1444_p3;
wire   [0:0] icmp_ln1494_47_fu_1456_p2;
wire   [30:0] trunc_ln83_47_fu_1462_p1;
wire   [30:0] select_ln83_47_fu_1466_p3;
wire   [0:0] icmp_ln1494_48_fu_1478_p2;
wire   [30:0] trunc_ln83_48_fu_1484_p1;
wire   [30:0] select_ln83_48_fu_1488_p3;
wire   [0:0] icmp_ln1494_49_fu_1500_p2;
wire   [30:0] trunc_ln83_49_fu_1506_p1;
wire   [30:0] select_ln83_49_fu_1510_p3;
wire   [31:0] zext_ln83_fu_440_p1;
wire   [31:0] zext_ln83_1_fu_462_p1;
wire   [31:0] zext_ln83_2_fu_484_p1;
wire   [31:0] zext_ln83_3_fu_506_p1;
wire   [31:0] zext_ln83_4_fu_528_p1;
wire   [31:0] zext_ln83_5_fu_550_p1;
wire   [31:0] zext_ln83_6_fu_572_p1;
wire   [31:0] zext_ln83_7_fu_594_p1;
wire   [31:0] zext_ln83_8_fu_616_p1;
wire   [31:0] zext_ln83_9_fu_638_p1;
wire   [31:0] zext_ln83_10_fu_660_p1;
wire   [31:0] zext_ln83_11_fu_682_p1;
wire   [31:0] zext_ln83_12_fu_704_p1;
wire   [31:0] zext_ln83_13_fu_726_p1;
wire   [31:0] zext_ln83_14_fu_748_p1;
wire   [31:0] zext_ln83_15_fu_770_p1;
wire   [31:0] zext_ln83_16_fu_792_p1;
wire   [31:0] zext_ln83_17_fu_814_p1;
wire   [31:0] zext_ln83_18_fu_836_p1;
wire   [31:0] zext_ln83_19_fu_858_p1;
wire   [31:0] zext_ln83_20_fu_880_p1;
wire   [31:0] zext_ln83_21_fu_902_p1;
wire   [31:0] zext_ln83_22_fu_924_p1;
wire   [31:0] zext_ln83_23_fu_946_p1;
wire   [31:0] zext_ln83_24_fu_968_p1;
wire   [31:0] zext_ln83_25_fu_990_p1;
wire   [31:0] zext_ln83_26_fu_1012_p1;
wire   [31:0] zext_ln83_27_fu_1034_p1;
wire   [31:0] zext_ln83_28_fu_1056_p1;
wire   [31:0] zext_ln83_29_fu_1078_p1;
wire   [31:0] zext_ln83_30_fu_1100_p1;
wire   [31:0] zext_ln83_31_fu_1122_p1;
wire   [31:0] zext_ln83_32_fu_1144_p1;
wire   [31:0] zext_ln83_33_fu_1166_p1;
wire   [31:0] zext_ln83_34_fu_1188_p1;
wire   [31:0] zext_ln83_35_fu_1210_p1;
wire   [31:0] zext_ln83_36_fu_1232_p1;
wire   [31:0] zext_ln83_37_fu_1254_p1;
wire   [31:0] zext_ln83_38_fu_1276_p1;
wire   [31:0] zext_ln83_39_fu_1298_p1;
wire   [31:0] zext_ln83_40_fu_1320_p1;
wire   [31:0] zext_ln83_41_fu_1342_p1;
wire   [31:0] zext_ln83_42_fu_1364_p1;
wire   [31:0] zext_ln83_43_fu_1386_p1;
wire   [31:0] zext_ln83_44_fu_1408_p1;
wire   [31:0] zext_ln83_45_fu_1430_p1;
wire   [31:0] zext_ln83_46_fu_1452_p1;
wire   [31:0] zext_ln83_47_fu_1474_p1;
wire   [31:0] zext_ln83_48_fu_1496_p1;
wire   [31:0] zext_ln83_49_fu_1518_p1;

assign ap_ready = 1'b1;

assign ap_return_0 = zext_ln83_fu_440_p1;

assign ap_return_1 = zext_ln83_1_fu_462_p1;

assign ap_return_10 = zext_ln83_10_fu_660_p1;

assign ap_return_11 = zext_ln83_11_fu_682_p1;

assign ap_return_12 = zext_ln83_12_fu_704_p1;

assign ap_return_13 = zext_ln83_13_fu_726_p1;

assign ap_return_14 = zext_ln83_14_fu_748_p1;

assign ap_return_15 = zext_ln83_15_fu_770_p1;

assign ap_return_16 = zext_ln83_16_fu_792_p1;

assign ap_return_17 = zext_ln83_17_fu_814_p1;

assign ap_return_18 = zext_ln83_18_fu_836_p1;

assign ap_return_19 = zext_ln83_19_fu_858_p1;

assign ap_return_2 = zext_ln83_2_fu_484_p1;

assign ap_return_20 = zext_ln83_20_fu_880_p1;

assign ap_return_21 = zext_ln83_21_fu_902_p1;

assign ap_return_22 = zext_ln83_22_fu_924_p1;

assign ap_return_23 = zext_ln83_23_fu_946_p1;

assign ap_return_24 = zext_ln83_24_fu_968_p1;

assign ap_return_25 = zext_ln83_25_fu_990_p1;

assign ap_return_26 = zext_ln83_26_fu_1012_p1;

assign ap_return_27 = zext_ln83_27_fu_1034_p1;

assign ap_return_28 = zext_ln83_28_fu_1056_p1;

assign ap_return_29 = zext_ln83_29_fu_1078_p1;

assign ap_return_3 = zext_ln83_3_fu_506_p1;

assign ap_return_30 = zext_ln83_30_fu_1100_p1;

assign ap_return_31 = zext_ln83_31_fu_1122_p1;

assign ap_return_32 = zext_ln83_32_fu_1144_p1;

assign ap_return_33 = zext_ln83_33_fu_1166_p1;

assign ap_return_34 = zext_ln83_34_fu_1188_p1;

assign ap_return_35 = zext_ln83_35_fu_1210_p1;

assign ap_return_36 = zext_ln83_36_fu_1232_p1;

assign ap_return_37 = zext_ln83_37_fu_1254_p1;

assign ap_return_38 = zext_ln83_38_fu_1276_p1;

assign ap_return_39 = zext_ln83_39_fu_1298_p1;

assign ap_return_4 = zext_ln83_4_fu_528_p1;

assign ap_return_40 = zext_ln83_40_fu_1320_p1;

assign ap_return_41 = zext_ln83_41_fu_1342_p1;

assign ap_return_42 = zext_ln83_42_fu_1364_p1;

assign ap_return_43 = zext_ln83_43_fu_1386_p1;

assign ap_return_44 = zext_ln83_44_fu_1408_p1;

assign ap_return_45 = zext_ln83_45_fu_1430_p1;

assign ap_return_46 = zext_ln83_46_fu_1452_p1;

assign ap_return_47 = zext_ln83_47_fu_1474_p1;

assign ap_return_48 = zext_ln83_48_fu_1496_p1;

assign ap_return_49 = zext_ln83_49_fu_1518_p1;

assign ap_return_5 = zext_ln83_5_fu_550_p1;

assign ap_return_6 = zext_ln83_6_fu_572_p1;

assign ap_return_7 = zext_ln83_7_fu_594_p1;

assign ap_return_8 = zext_ln83_8_fu_616_p1;

assign ap_return_9 = zext_ln83_9_fu_638_p1;

assign icmp_ln1494_10_fu_642_p2 = (($signed(data_10_V_read) > $signed(32'd0)) ? 1'b1 : 1'b0);

assign icmp_ln1494_11_fu_664_p2 = (($signed(data_11_V_read) > $signed(32'd0)) ? 1'b1 : 1'b0);

assign icmp_ln1494_12_fu_686_p2 = (($signed(data_12_V_read) > $signed(32'd0)) ? 1'b1 : 1'b0);

assign icmp_ln1494_13_fu_708_p2 = (($signed(data_13_V_read) > $signed(32'd0)) ? 1'b1 : 1'b0);

assign icmp_ln1494_14_fu_730_p2 = (($signed(data_14_V_read) > $signed(32'd0)) ? 1'b1 : 1'b0);

assign icmp_ln1494_15_fu_752_p2 = (($signed(data_15_V_read) > $signed(32'd0)) ? 1'b1 : 1'b0);

assign icmp_ln1494_16_fu_774_p2 = (($signed(data_16_V_read) > $signed(32'd0)) ? 1'b1 : 1'b0);

assign icmp_ln1494_17_fu_796_p2 = (($signed(data_17_V_read) > $signed(32'd0)) ? 1'b1 : 1'b0);

assign icmp_ln1494_18_fu_818_p2 = (($signed(data_18_V_read) > $signed(32'd0)) ? 1'b1 : 1'b0);

assign icmp_ln1494_19_fu_840_p2 = (($signed(data_19_V_read) > $signed(32'd0)) ? 1'b1 : 1'b0);

assign icmp_ln1494_1_fu_444_p2 = (($signed(data_1_V_read) > $signed(32'd0)) ? 1'b1 : 1'b0);

assign icmp_ln1494_20_fu_862_p2 = (($signed(data_20_V_read) > $signed(32'd0)) ? 1'b1 : 1'b0);

assign icmp_ln1494_21_fu_884_p2 = (($signed(data_21_V_read) > $signed(32'd0)) ? 1'b1 : 1'b0);

assign icmp_ln1494_22_fu_906_p2 = (($signed(data_22_V_read) > $signed(32'd0)) ? 1'b1 : 1'b0);

assign icmp_ln1494_23_fu_928_p2 = (($signed(data_23_V_read) > $signed(32'd0)) ? 1'b1 : 1'b0);

assign icmp_ln1494_24_fu_950_p2 = (($signed(data_24_V_read) > $signed(32'd0)) ? 1'b1 : 1'b0);

assign icmp_ln1494_25_fu_972_p2 = (($signed(data_25_V_read) > $signed(32'd0)) ? 1'b1 : 1'b0);

assign icmp_ln1494_26_fu_994_p2 = (($signed(data_26_V_read) > $signed(32'd0)) ? 1'b1 : 1'b0);

assign icmp_ln1494_27_fu_1016_p2 = (($signed(data_27_V_read) > $signed(32'd0)) ? 1'b1 : 1'b0);

assign icmp_ln1494_28_fu_1038_p2 = (($signed(data_28_V_read) > $signed(32'd0)) ? 1'b1 : 1'b0);

assign icmp_ln1494_29_fu_1060_p2 = (($signed(data_29_V_read) > $signed(32'd0)) ? 1'b1 : 1'b0);

assign icmp_ln1494_2_fu_466_p2 = (($signed(data_2_V_read) > $signed(32'd0)) ? 1'b1 : 1'b0);

assign icmp_ln1494_30_fu_1082_p2 = (($signed(data_30_V_read) > $signed(32'd0)) ? 1'b1 : 1'b0);

assign icmp_ln1494_31_fu_1104_p2 = (($signed(data_31_V_read) > $signed(32'd0)) ? 1'b1 : 1'b0);

assign icmp_ln1494_32_fu_1126_p2 = (($signed(data_32_V_read) > $signed(32'd0)) ? 1'b1 : 1'b0);

assign icmp_ln1494_33_fu_1148_p2 = (($signed(data_33_V_read) > $signed(32'd0)) ? 1'b1 : 1'b0);

assign icmp_ln1494_34_fu_1170_p2 = (($signed(data_34_V_read) > $signed(32'd0)) ? 1'b1 : 1'b0);

assign icmp_ln1494_35_fu_1192_p2 = (($signed(data_35_V_read) > $signed(32'd0)) ? 1'b1 : 1'b0);

assign icmp_ln1494_36_fu_1214_p2 = (($signed(data_36_V_read) > $signed(32'd0)) ? 1'b1 : 1'b0);

assign icmp_ln1494_37_fu_1236_p2 = (($signed(data_37_V_read) > $signed(32'd0)) ? 1'b1 : 1'b0);

assign icmp_ln1494_38_fu_1258_p2 = (($signed(data_38_V_read) > $signed(32'd0)) ? 1'b1 : 1'b0);

assign icmp_ln1494_39_fu_1280_p2 = (($signed(data_39_V_read) > $signed(32'd0)) ? 1'b1 : 1'b0);

assign icmp_ln1494_3_fu_488_p2 = (($signed(data_3_V_read) > $signed(32'd0)) ? 1'b1 : 1'b0);

assign icmp_ln1494_40_fu_1302_p2 = (($signed(data_40_V_read) > $signed(32'd0)) ? 1'b1 : 1'b0);

assign icmp_ln1494_41_fu_1324_p2 = (($signed(data_41_V_read) > $signed(32'd0)) ? 1'b1 : 1'b0);

assign icmp_ln1494_42_fu_1346_p2 = (($signed(data_42_V_read) > $signed(32'd0)) ? 1'b1 : 1'b0);

assign icmp_ln1494_43_fu_1368_p2 = (($signed(data_43_V_read) > $signed(32'd0)) ? 1'b1 : 1'b0);

assign icmp_ln1494_44_fu_1390_p2 = (($signed(data_44_V_read) > $signed(32'd0)) ? 1'b1 : 1'b0);

assign icmp_ln1494_45_fu_1412_p2 = (($signed(data_45_V_read) > $signed(32'd0)) ? 1'b1 : 1'b0);

assign icmp_ln1494_46_fu_1434_p2 = (($signed(data_46_V_read) > $signed(32'd0)) ? 1'b1 : 1'b0);

assign icmp_ln1494_47_fu_1456_p2 = (($signed(data_47_V_read) > $signed(32'd0)) ? 1'b1 : 1'b0);

assign icmp_ln1494_48_fu_1478_p2 = (($signed(data_48_V_read) > $signed(32'd0)) ? 1'b1 : 1'b0);

assign icmp_ln1494_49_fu_1500_p2 = (($signed(data_49_V_read) > $signed(32'd0)) ? 1'b1 : 1'b0);

assign icmp_ln1494_4_fu_510_p2 = (($signed(data_4_V_read) > $signed(32'd0)) ? 1'b1 : 1'b0);

assign icmp_ln1494_5_fu_532_p2 = (($signed(data_5_V_read) > $signed(32'd0)) ? 1'b1 : 1'b0);

assign icmp_ln1494_6_fu_554_p2 = (($signed(data_6_V_read) > $signed(32'd0)) ? 1'b1 : 1'b0);

assign icmp_ln1494_7_fu_576_p2 = (($signed(data_7_V_read) > $signed(32'd0)) ? 1'b1 : 1'b0);

assign icmp_ln1494_8_fu_598_p2 = (($signed(data_8_V_read) > $signed(32'd0)) ? 1'b1 : 1'b0);

assign icmp_ln1494_9_fu_620_p2 = (($signed(data_9_V_read) > $signed(32'd0)) ? 1'b1 : 1'b0);

assign icmp_ln1494_fu_422_p2 = (($signed(data_0_V_read) > $signed(32'd0)) ? 1'b1 : 1'b0);

assign select_ln83_10_fu_652_p3 = ((icmp_ln1494_10_fu_642_p2[0:0] === 1'b1) ? trunc_ln83_10_fu_648_p1 : 31'd0);

assign select_ln83_11_fu_674_p3 = ((icmp_ln1494_11_fu_664_p2[0:0] === 1'b1) ? trunc_ln83_11_fu_670_p1 : 31'd0);

assign select_ln83_12_fu_696_p3 = ((icmp_ln1494_12_fu_686_p2[0:0] === 1'b1) ? trunc_ln83_12_fu_692_p1 : 31'd0);

assign select_ln83_13_fu_718_p3 = ((icmp_ln1494_13_fu_708_p2[0:0] === 1'b1) ? trunc_ln83_13_fu_714_p1 : 31'd0);

assign select_ln83_14_fu_740_p3 = ((icmp_ln1494_14_fu_730_p2[0:0] === 1'b1) ? trunc_ln83_14_fu_736_p1 : 31'd0);

assign select_ln83_15_fu_762_p3 = ((icmp_ln1494_15_fu_752_p2[0:0] === 1'b1) ? trunc_ln83_15_fu_758_p1 : 31'd0);

assign select_ln83_16_fu_784_p3 = ((icmp_ln1494_16_fu_774_p2[0:0] === 1'b1) ? trunc_ln83_16_fu_780_p1 : 31'd0);

assign select_ln83_17_fu_806_p3 = ((icmp_ln1494_17_fu_796_p2[0:0] === 1'b1) ? trunc_ln83_17_fu_802_p1 : 31'd0);

assign select_ln83_18_fu_828_p3 = ((icmp_ln1494_18_fu_818_p2[0:0] === 1'b1) ? trunc_ln83_18_fu_824_p1 : 31'd0);

assign select_ln83_19_fu_850_p3 = ((icmp_ln1494_19_fu_840_p2[0:0] === 1'b1) ? trunc_ln83_19_fu_846_p1 : 31'd0);

assign select_ln83_1_fu_454_p3 = ((icmp_ln1494_1_fu_444_p2[0:0] === 1'b1) ? trunc_ln83_1_fu_450_p1 : 31'd0);

assign select_ln83_20_fu_872_p3 = ((icmp_ln1494_20_fu_862_p2[0:0] === 1'b1) ? trunc_ln83_20_fu_868_p1 : 31'd0);

assign select_ln83_21_fu_894_p3 = ((icmp_ln1494_21_fu_884_p2[0:0] === 1'b1) ? trunc_ln83_21_fu_890_p1 : 31'd0);

assign select_ln83_22_fu_916_p3 = ((icmp_ln1494_22_fu_906_p2[0:0] === 1'b1) ? trunc_ln83_22_fu_912_p1 : 31'd0);

assign select_ln83_23_fu_938_p3 = ((icmp_ln1494_23_fu_928_p2[0:0] === 1'b1) ? trunc_ln83_23_fu_934_p1 : 31'd0);

assign select_ln83_24_fu_960_p3 = ((icmp_ln1494_24_fu_950_p2[0:0] === 1'b1) ? trunc_ln83_24_fu_956_p1 : 31'd0);

assign select_ln83_25_fu_982_p3 = ((icmp_ln1494_25_fu_972_p2[0:0] === 1'b1) ? trunc_ln83_25_fu_978_p1 : 31'd0);

assign select_ln83_26_fu_1004_p3 = ((icmp_ln1494_26_fu_994_p2[0:0] === 1'b1) ? trunc_ln83_26_fu_1000_p1 : 31'd0);

assign select_ln83_27_fu_1026_p3 = ((icmp_ln1494_27_fu_1016_p2[0:0] === 1'b1) ? trunc_ln83_27_fu_1022_p1 : 31'd0);

assign select_ln83_28_fu_1048_p3 = ((icmp_ln1494_28_fu_1038_p2[0:0] === 1'b1) ? trunc_ln83_28_fu_1044_p1 : 31'd0);

assign select_ln83_29_fu_1070_p3 = ((icmp_ln1494_29_fu_1060_p2[0:0] === 1'b1) ? trunc_ln83_29_fu_1066_p1 : 31'd0);

assign select_ln83_2_fu_476_p3 = ((icmp_ln1494_2_fu_466_p2[0:0] === 1'b1) ? trunc_ln83_2_fu_472_p1 : 31'd0);

assign select_ln83_30_fu_1092_p3 = ((icmp_ln1494_30_fu_1082_p2[0:0] === 1'b1) ? trunc_ln83_30_fu_1088_p1 : 31'd0);

assign select_ln83_31_fu_1114_p3 = ((icmp_ln1494_31_fu_1104_p2[0:0] === 1'b1) ? trunc_ln83_31_fu_1110_p1 : 31'd0);

assign select_ln83_32_fu_1136_p3 = ((icmp_ln1494_32_fu_1126_p2[0:0] === 1'b1) ? trunc_ln83_32_fu_1132_p1 : 31'd0);

assign select_ln83_33_fu_1158_p3 = ((icmp_ln1494_33_fu_1148_p2[0:0] === 1'b1) ? trunc_ln83_33_fu_1154_p1 : 31'd0);

assign select_ln83_34_fu_1180_p3 = ((icmp_ln1494_34_fu_1170_p2[0:0] === 1'b1) ? trunc_ln83_34_fu_1176_p1 : 31'd0);

assign select_ln83_35_fu_1202_p3 = ((icmp_ln1494_35_fu_1192_p2[0:0] === 1'b1) ? trunc_ln83_35_fu_1198_p1 : 31'd0);

assign select_ln83_36_fu_1224_p3 = ((icmp_ln1494_36_fu_1214_p2[0:0] === 1'b1) ? trunc_ln83_36_fu_1220_p1 : 31'd0);

assign select_ln83_37_fu_1246_p3 = ((icmp_ln1494_37_fu_1236_p2[0:0] === 1'b1) ? trunc_ln83_37_fu_1242_p1 : 31'd0);

assign select_ln83_38_fu_1268_p3 = ((icmp_ln1494_38_fu_1258_p2[0:0] === 1'b1) ? trunc_ln83_38_fu_1264_p1 : 31'd0);

assign select_ln83_39_fu_1290_p3 = ((icmp_ln1494_39_fu_1280_p2[0:0] === 1'b1) ? trunc_ln83_39_fu_1286_p1 : 31'd0);

assign select_ln83_3_fu_498_p3 = ((icmp_ln1494_3_fu_488_p2[0:0] === 1'b1) ? trunc_ln83_3_fu_494_p1 : 31'd0);

assign select_ln83_40_fu_1312_p3 = ((icmp_ln1494_40_fu_1302_p2[0:0] === 1'b1) ? trunc_ln83_40_fu_1308_p1 : 31'd0);

assign select_ln83_41_fu_1334_p3 = ((icmp_ln1494_41_fu_1324_p2[0:0] === 1'b1) ? trunc_ln83_41_fu_1330_p1 : 31'd0);

assign select_ln83_42_fu_1356_p3 = ((icmp_ln1494_42_fu_1346_p2[0:0] === 1'b1) ? trunc_ln83_42_fu_1352_p1 : 31'd0);

assign select_ln83_43_fu_1378_p3 = ((icmp_ln1494_43_fu_1368_p2[0:0] === 1'b1) ? trunc_ln83_43_fu_1374_p1 : 31'd0);

assign select_ln83_44_fu_1400_p3 = ((icmp_ln1494_44_fu_1390_p2[0:0] === 1'b1) ? trunc_ln83_44_fu_1396_p1 : 31'd0);

assign select_ln83_45_fu_1422_p3 = ((icmp_ln1494_45_fu_1412_p2[0:0] === 1'b1) ? trunc_ln83_45_fu_1418_p1 : 31'd0);

assign select_ln83_46_fu_1444_p3 = ((icmp_ln1494_46_fu_1434_p2[0:0] === 1'b1) ? trunc_ln83_46_fu_1440_p1 : 31'd0);

assign select_ln83_47_fu_1466_p3 = ((icmp_ln1494_47_fu_1456_p2[0:0] === 1'b1) ? trunc_ln83_47_fu_1462_p1 : 31'd0);

assign select_ln83_48_fu_1488_p3 = ((icmp_ln1494_48_fu_1478_p2[0:0] === 1'b1) ? trunc_ln83_48_fu_1484_p1 : 31'd0);

assign select_ln83_49_fu_1510_p3 = ((icmp_ln1494_49_fu_1500_p2[0:0] === 1'b1) ? trunc_ln83_49_fu_1506_p1 : 31'd0);

assign select_ln83_4_fu_520_p3 = ((icmp_ln1494_4_fu_510_p2[0:0] === 1'b1) ? trunc_ln83_4_fu_516_p1 : 31'd0);

assign select_ln83_5_fu_542_p3 = ((icmp_ln1494_5_fu_532_p2[0:0] === 1'b1) ? trunc_ln83_5_fu_538_p1 : 31'd0);

assign select_ln83_6_fu_564_p3 = ((icmp_ln1494_6_fu_554_p2[0:0] === 1'b1) ? trunc_ln83_6_fu_560_p1 : 31'd0);

assign select_ln83_7_fu_586_p3 = ((icmp_ln1494_7_fu_576_p2[0:0] === 1'b1) ? trunc_ln83_7_fu_582_p1 : 31'd0);

assign select_ln83_8_fu_608_p3 = ((icmp_ln1494_8_fu_598_p2[0:0] === 1'b1) ? trunc_ln83_8_fu_604_p1 : 31'd0);

assign select_ln83_9_fu_630_p3 = ((icmp_ln1494_9_fu_620_p2[0:0] === 1'b1) ? trunc_ln83_9_fu_626_p1 : 31'd0);

assign select_ln83_fu_432_p3 = ((icmp_ln1494_fu_422_p2[0:0] === 1'b1) ? trunc_ln83_fu_428_p1 : 31'd0);

assign trunc_ln83_10_fu_648_p1 = data_10_V_read[30:0];

assign trunc_ln83_11_fu_670_p1 = data_11_V_read[30:0];

assign trunc_ln83_12_fu_692_p1 = data_12_V_read[30:0];

assign trunc_ln83_13_fu_714_p1 = data_13_V_read[30:0];

assign trunc_ln83_14_fu_736_p1 = data_14_V_read[30:0];

assign trunc_ln83_15_fu_758_p1 = data_15_V_read[30:0];

assign trunc_ln83_16_fu_780_p1 = data_16_V_read[30:0];

assign trunc_ln83_17_fu_802_p1 = data_17_V_read[30:0];

assign trunc_ln83_18_fu_824_p1 = data_18_V_read[30:0];

assign trunc_ln83_19_fu_846_p1 = data_19_V_read[30:0];

assign trunc_ln83_1_fu_450_p1 = data_1_V_read[30:0];

assign trunc_ln83_20_fu_868_p1 = data_20_V_read[30:0];

assign trunc_ln83_21_fu_890_p1 = data_21_V_read[30:0];

assign trunc_ln83_22_fu_912_p1 = data_22_V_read[30:0];

assign trunc_ln83_23_fu_934_p1 = data_23_V_read[30:0];

assign trunc_ln83_24_fu_956_p1 = data_24_V_read[30:0];

assign trunc_ln83_25_fu_978_p1 = data_25_V_read[30:0];

assign trunc_ln83_26_fu_1000_p1 = data_26_V_read[30:0];

assign trunc_ln83_27_fu_1022_p1 = data_27_V_read[30:0];

assign trunc_ln83_28_fu_1044_p1 = data_28_V_read[30:0];

assign trunc_ln83_29_fu_1066_p1 = data_29_V_read[30:0];

assign trunc_ln83_2_fu_472_p1 = data_2_V_read[30:0];

assign trunc_ln83_30_fu_1088_p1 = data_30_V_read[30:0];

assign trunc_ln83_31_fu_1110_p1 = data_31_V_read[30:0];

assign trunc_ln83_32_fu_1132_p1 = data_32_V_read[30:0];

assign trunc_ln83_33_fu_1154_p1 = data_33_V_read[30:0];

assign trunc_ln83_34_fu_1176_p1 = data_34_V_read[30:0];

assign trunc_ln83_35_fu_1198_p1 = data_35_V_read[30:0];

assign trunc_ln83_36_fu_1220_p1 = data_36_V_read[30:0];

assign trunc_ln83_37_fu_1242_p1 = data_37_V_read[30:0];

assign trunc_ln83_38_fu_1264_p1 = data_38_V_read[30:0];

assign trunc_ln83_39_fu_1286_p1 = data_39_V_read[30:0];

assign trunc_ln83_3_fu_494_p1 = data_3_V_read[30:0];

assign trunc_ln83_40_fu_1308_p1 = data_40_V_read[30:0];

assign trunc_ln83_41_fu_1330_p1 = data_41_V_read[30:0];

assign trunc_ln83_42_fu_1352_p1 = data_42_V_read[30:0];

assign trunc_ln83_43_fu_1374_p1 = data_43_V_read[30:0];

assign trunc_ln83_44_fu_1396_p1 = data_44_V_read[30:0];

assign trunc_ln83_45_fu_1418_p1 = data_45_V_read[30:0];

assign trunc_ln83_46_fu_1440_p1 = data_46_V_read[30:0];

assign trunc_ln83_47_fu_1462_p1 = data_47_V_read[30:0];

assign trunc_ln83_48_fu_1484_p1 = data_48_V_read[30:0];

assign trunc_ln83_49_fu_1506_p1 = data_49_V_read[30:0];

assign trunc_ln83_4_fu_516_p1 = data_4_V_read[30:0];

assign trunc_ln83_5_fu_538_p1 = data_5_V_read[30:0];

assign trunc_ln83_6_fu_560_p1 = data_6_V_read[30:0];

assign trunc_ln83_7_fu_582_p1 = data_7_V_read[30:0];

assign trunc_ln83_8_fu_604_p1 = data_8_V_read[30:0];

assign trunc_ln83_9_fu_626_p1 = data_9_V_read[30:0];

assign trunc_ln83_fu_428_p1 = data_0_V_read[30:0];

assign zext_ln83_10_fu_660_p1 = select_ln83_10_fu_652_p3;

assign zext_ln83_11_fu_682_p1 = select_ln83_11_fu_674_p3;

assign zext_ln83_12_fu_704_p1 = select_ln83_12_fu_696_p3;

assign zext_ln83_13_fu_726_p1 = select_ln83_13_fu_718_p3;

assign zext_ln83_14_fu_748_p1 = select_ln83_14_fu_740_p3;

assign zext_ln83_15_fu_770_p1 = select_ln83_15_fu_762_p3;

assign zext_ln83_16_fu_792_p1 = select_ln83_16_fu_784_p3;

assign zext_ln83_17_fu_814_p1 = select_ln83_17_fu_806_p3;

assign zext_ln83_18_fu_836_p1 = select_ln83_18_fu_828_p3;

assign zext_ln83_19_fu_858_p1 = select_ln83_19_fu_850_p3;

assign zext_ln83_1_fu_462_p1 = select_ln83_1_fu_454_p3;

assign zext_ln83_20_fu_880_p1 = select_ln83_20_fu_872_p3;

assign zext_ln83_21_fu_902_p1 = select_ln83_21_fu_894_p3;

assign zext_ln83_22_fu_924_p1 = select_ln83_22_fu_916_p3;

assign zext_ln83_23_fu_946_p1 = select_ln83_23_fu_938_p3;

assign zext_ln83_24_fu_968_p1 = select_ln83_24_fu_960_p3;

assign zext_ln83_25_fu_990_p1 = select_ln83_25_fu_982_p3;

assign zext_ln83_26_fu_1012_p1 = select_ln83_26_fu_1004_p3;

assign zext_ln83_27_fu_1034_p1 = select_ln83_27_fu_1026_p3;

assign zext_ln83_28_fu_1056_p1 = select_ln83_28_fu_1048_p3;

assign zext_ln83_29_fu_1078_p1 = select_ln83_29_fu_1070_p3;

assign zext_ln83_2_fu_484_p1 = select_ln83_2_fu_476_p3;

assign zext_ln83_30_fu_1100_p1 = select_ln83_30_fu_1092_p3;

assign zext_ln83_31_fu_1122_p1 = select_ln83_31_fu_1114_p3;

assign zext_ln83_32_fu_1144_p1 = select_ln83_32_fu_1136_p3;

assign zext_ln83_33_fu_1166_p1 = select_ln83_33_fu_1158_p3;

assign zext_ln83_34_fu_1188_p1 = select_ln83_34_fu_1180_p3;

assign zext_ln83_35_fu_1210_p1 = select_ln83_35_fu_1202_p3;

assign zext_ln83_36_fu_1232_p1 = select_ln83_36_fu_1224_p3;

assign zext_ln83_37_fu_1254_p1 = select_ln83_37_fu_1246_p3;

assign zext_ln83_38_fu_1276_p1 = select_ln83_38_fu_1268_p3;

assign zext_ln83_39_fu_1298_p1 = select_ln83_39_fu_1290_p3;

assign zext_ln83_3_fu_506_p1 = select_ln83_3_fu_498_p3;

assign zext_ln83_40_fu_1320_p1 = select_ln83_40_fu_1312_p3;

assign zext_ln83_41_fu_1342_p1 = select_ln83_41_fu_1334_p3;

assign zext_ln83_42_fu_1364_p1 = select_ln83_42_fu_1356_p3;

assign zext_ln83_43_fu_1386_p1 = select_ln83_43_fu_1378_p3;

assign zext_ln83_44_fu_1408_p1 = select_ln83_44_fu_1400_p3;

assign zext_ln83_45_fu_1430_p1 = select_ln83_45_fu_1422_p3;

assign zext_ln83_46_fu_1452_p1 = select_ln83_46_fu_1444_p3;

assign zext_ln83_47_fu_1474_p1 = select_ln83_47_fu_1466_p3;

assign zext_ln83_48_fu_1496_p1 = select_ln83_48_fu_1488_p3;

assign zext_ln83_49_fu_1518_p1 = select_ln83_49_fu_1510_p3;

assign zext_ln83_4_fu_528_p1 = select_ln83_4_fu_520_p3;

assign zext_ln83_5_fu_550_p1 = select_ln83_5_fu_542_p3;

assign zext_ln83_6_fu_572_p1 = select_ln83_6_fu_564_p3;

assign zext_ln83_7_fu_594_p1 = select_ln83_7_fu_586_p3;

assign zext_ln83_8_fu_616_p1 = select_ln83_8_fu_608_p3;

assign zext_ln83_9_fu_638_p1 = select_ln83_9_fu_630_p3;

assign zext_ln83_fu_440_p1 = select_ln83_fu_432_p3;

endmodule //relu_1