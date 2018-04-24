import numpy as np
import math
import pandas as pd
from numpy.core.multiarray import dtype
from sklearn.metrics import confusion_matrix
import operator


class Bayesian:
    mean = np.genfromtxt('means.csv')
    sigma = np.array([
    [0.000994,0.001194,0.000969,0.000017,0.019485,0.000027,0.000009,0.000447,0.001054,0.000497,0.000434,0.000145,0.002438,0.000888,0.001067,0.001798,0.001012,0.001354,0.000569,0.001188,0.001107,0.001162,0.000705,0.000790,0.000705,0.000756,0.001514,0.001032,0.001180,0.000968,0.000929,0.000936,0.001184,0.001314,0.000488,0.000867,0.001037,0.000559,0.000676,0.001236,0.000955,0.001401,0.000817,0.001242,0.001096,0.001252,0.000745,0.001276,0.000947,0.001118,0.000370,0.001190,0.001032,0.001063,0.000734,0.000810,0.001568,0.000733,0.000941,0.001396,0.001135,0.000538,0.001021,0.001359,0.000835,0.000819,0.001166,0.000568,0.000908,0.001019,0.001363],
    [0.000564,0.000264,0.000170,0.000001,0.000794,0.000002,0.000004,0.000197,0.000685,0.000383,0.000355,0.000094,0.000991,0.000924,0.000754,0.001026,0.000356,0.000628,0.000506,0.001097,0.000653,0.000992,0.000845,0.000523,0.000662,0.000558,0.000997,0.000886,0.000935,0.000809,0.000395,0.000819,0.000892,0.000502,0.000704,0.000817,0.000599,0.000334,0.001265,0.001490,0.000686,0.000843,0.000836,0.000910,0.001144,0.000594,0.000607,0.000712,0.000557,0.000901,0.000715,0.000407,0.000873,0.000814,0.000618,0.000913,0.001333,0.000990,0.000792,0.001320,0.001044,0.000484,0.001497,0.000519,0.000617,0.001203,0.000839,0.000608,0.000503,0.000880,0.001152],
    [0.000993,0.000749,0.000511,0.000005,0.001766,0.000006,0.000002,0.000210,0.000129,0.000196,0.000213,0.000018,0.002464,0.000590,0.001063,0.001154,0.001066,0.001707,0.001088,0.000983,0.000862,0.001014,0.000540,0.001001,0.001354,0.000695,0.000533,0.000471,0.001255,0.001032,0.001154,0.000546,0.000300,0.001148,0.000259,0.001168,0.000199,0.000101,0.001180,0.001270,0.000234,0.000424,0.001373,0.000763,0.001917,0.001289,0.000658,0.001164,0.000105,0.000717,0.000563,0.000715,0.001180,0.000188,0.001255,0.001468,0.001280,0.000538,0.000999,0.001564,0.000509,0.000209,0.001018,0.000365,0.000333,0.000213,0.000719,0.000928,0.001650,0.001327,0.000279],
    [0.000442,0.000403,0.000257,0.000002,0.000536,0.000003,0.000004,0.000197,0.000593,0.000281,0.000248,0.000081,0.000917,0.000855,0.000933,0.001364,0.000882,0.000622,0.000801,0.000997,0.001202,0.001368,0.000872,0.000775,0.000550,0.000796,0.000927,0.000459,0.001024,0.000751,0.000459,0.000643,0.000805,0.000524,0.000547,0.000978,0.000337,0.000361,0.000761,0.000725,0.000555,0.000777,0.001160,0.000994,0.000841,0.000659,0.000776,0.000742,0.000763,0.000931,0.000396,0.000501,0.000692,0.000439,0.000424,0.000550,0.000880,0.000769,0.000618,0.000524,0.000568,0.000666,0.000983,0.000667,0.000820,0.000619,0.000912,0.000503,0.000984,0.000866,0.001166],
    [0.000401,0.000284,0.000195,0.000002,0.001116,0.000002,0.000002,0.000110,0.000332,0.000060,0.000054,0.000046,0.000851,0.000571,0.000431,0.000429,0.000521,0.000524,0.000681,0.000968,0.000619,0.000767,0.001034,0.000896,0.000886,0.000897,0.001686,0.000762,0.001030,0.000798,0.000898,0.000875,0.000646,0.000864,0.000463,0.000940,0.001479,0.000364,0.001250,0.000697,0.001034,0.001341,0.000819,0.000761,0.001136,0.001262,0.000726,0.001364,0.001110,0.000961,0.001333,0.000987,0.001479,0.000696,0.000452,0.000994,0.001470,0.000807,0.001543,0.000687,0.000852,0.000923,0.001275,0.000535,0.001598,0.001543,0.001855,0.001047,0.001110,0.001236,0.000981],
    [0.000352,0.000351,0.000244,0.000002,0.000452,0.000003,0.000003,0.000209,0.000697,0.000303,0.000276,0.000096,0.000703,0.000840,0.000647,0.000817,0.000563,0.000727,0.000987,0.000379,0.000702,0.001369,0.000868,0.000917,0.000904,0.000428,0.000751,0.000762,0.001562,0.000572,0.000795,0.000756,0.000670,0.000617,0.000601,0.000601,0.000641,0.000540,0.000788,0.000750,0.000932,0.000969,0.001172,0.000816,0.001161,0.000816,0.000692,0.000782,0.000369,0.001112,0.000698,0.000532,0.000850,0.000637,0.000510,0.000510,0.000935,0.001296,0.000974,0.000844,0.000628,0.000631,0.001004,0.000569,0.001248,0.001477,0.001176,0.000527,0.000791,0.000983,0.001476],
    [0.000486,0.000274,0.000174,0.000001,0.000806,0.000002,0.000002,0.000150,0.000512,0.000190,0.000173,0.000070,0.000963,0.000502,0.001202,0.001368,0.000771,0.001049,0.000795,0.001460,0.000824,0.000756,0.001341,0.000928,0.000898,0.000350,0.000571,0.000777,0.000877,0.000790,0.000481,0.000619,0.000672,0.000562,0.000525,0.000735,0.000415,0.000398,0.000685,0.001281,0.000759,0.000661,0.001256,0.000993,0.000595,0.000439,0.000742,0.000639,0.000553,0.001063,0.000506,0.000564,0.000360,0.000880,0.000805,0.000840,0.001046,0.001094,0.000547,0.000564,0.000516,0.000517,0.000605,0.000768,0.001025,0.000855,0.000891,0.001153,0.000882,0.000629,0.000859],
    [0.001858,0.001481,0.001155,0.000017,0.015804,0.000027,0.000015,0.000836,0.001563,0.000826,0.000690,0.000215,0.004112,0.000120,0.000730,0.001206,0.000513,0.001337,0.001004,0.000438,0.001095,0.000917,0.002229,0.001137,0.000707,0.001178,0.001044,0.000949,0.001274,0.000539,0.000539,0.001176,0.001451,0.001129,0.000557,0.001370,0.000571,0.000640,0.001720,0.000992,0.001511,0.001021,0.001528,0.000488,0.000192,0.001122,0.001056,0.001619,0.000719,0.000940,0.001204,0.000602,0.001548,0.000746,0.000320,0.001569,0.001228,0.000846,0.000252,0.001465,0.000656,0.000744,0.001669,0.001506,0.000735,0.001262,0.001226,0.000239,0.001694,0.001063,0.000989],
    [0.000294,0.000328,0.000217,0.000002,0.001235,0.000003,0.000002,0.000078,0.000580,0.000113,0.000092,0.000080,0.000649,0.000844,0.000734,0.001366,0.001222,0.001114,0.000741,0.001291,0.001445,0.001683,0.001538,0.000982,0.001353,0.001191,0.002364,0.002420,0.001836,0.001197,0.001558,0.001501,0.001338,0.001536,0.001118,0.001064,0.001235,0.000768,0.000820,0.001586,0.001475,0.001038,0.001331,0.001772,0.001407,0.001041,0.001470,0.001079,0.001214,0.001713,0.001271,0.000940,0.001501,0.001291,0.001415,0.001496,0.002273,0.000903,0.001187,0.001274,0.000624,0.000557,0.001326,0.001218,0.001921,0.001393,0.002013,0.001323,0.001058,0.001654,0.001505],
    [0.000317,0.000150,0.000103,0.000001,0.000892,0.000001,0.000003,0.000194,0.000259,0.000089,0.000090,0.000035,0.000466,0.000552,0.000325,0.000757,0.000780,0.000232,0.000548,0.000307,0.000475,0.001227,0.000654,0.000728,0.000369,0.000984,0.000901,0.000603,0.000271,0.000322,0.000880,0.000479,0.000608,0.000331,0.000382,0.001508,0.000266,0.000956,0.000690,0.000748,0.000284,0.000423,0.001028,0.000315,0.000721,0.000822,0.001223,0.000750,0.000422,0.000665,0.000363,0.000246,0.000507,0.000061,0.000213,0.000671,0.001394,0.000709,0.001005,0.001030,0.000981,0.000574,0.001072,0.000648,0.001023,0.000707,0.001700,0.000742,0.001195,0.002137,0.000893],
    [0.000481,0.000248,0.000155,0.000001,0.000525,0.000002,0.000001,0.000062,0.000883,0.000203,0.000163,0.000122,0.001023,0.000401,0.000554,0.001164,0.000300,0.000343,0.000130,0.000229,0.000718,0.000867,0.000119,0.000691,0.000439,0.000313,0.000162,0.000278,0.000100,0.000609,0.000740,0.000342,0.000612,0.000417,0.000426,0.001519,0.000532,0.000468,0.001150,0.000847,0.000051,0.000839,0.000836,0.000275,0.000822,0.000276,0.000069,0.000277,0.000828,0.000422,0.000212,0.000123,0.000105,0.000876,0.000059,0.000759,0.000745,0.000534,0.000681,0.001201,0.000589,0.001491,0.001879,0.000634,0.001625,0.001320,0.000938,0.000429,0.001377,0.001823,0.000978],
    [0.000099,0.000749,0.000508,0.000005,0.003539,0.000006,0.000003,0.000230,0.000264,0.000089,0.000082,0.000036,0.000159,0.000322,0.000322,0.001606,0.001338,0.000113,0.000442,0.000500,0.001464,0.001103,0.000669,0.000950,0.000416,0.001733,0.000149,0.000498,0.001132,0.001299,0.001823,0.000399,0.000976,0.002259,0.001209,0.001753,0.001317,0.000815,0.000939,0.003191,0.001130,0.001883,0.003773,0.001468,0.001158,0.001324,0.000786,0.000580,0.000315,0.000679,0.001801,0.000770,0.000459,0.000604,0.000776,0.002017,0.001585,0.001455,0.000484,0.000810,0.001033,0.000838,0.001103,0.001395,0.001528,0.001541,0.000170,0.000646,0.000761,0.000242,0.001151],
    [0.000412,0.000407,0.000291,0.000003,0.000989,0.000005,0.000002,0.000178,0.000695,0.000180,0.000150,0.000096,0.000957,0.000581,0.000505,0.000677,0.000457,0.001027,0.000450,0.000962,0.000936,0.000744,0.000483,0.000835,0.000737,0.000769,0.001117,0.000858,0.001230,0.000698,0.001074,0.000888,0.000933,0.000913,0.000422,0.000922,0.000726,0.000432,0.000766,0.000801,0.000422,0.000613,0.000915,0.000600,0.001031,0.001269,0.000927,0.001497,0.000696,0.001090,0.000571,0.000429,0.000596,0.000977,0.000592,0.000610,0.001623,0.000615,0.000515,0.000834,0.000683,0.000772,0.001542,0.001189,0.000725,0.001330,0.000932,0.000932,0.001139,0.001051,0.001372],
    [0.000416,0.000266,0.000175,0.000001,0.000711,0.000003,0.000005,0.000219,0.000732,0.000280,0.000242,0.000100,0.000777,0.000954,0.000891,0.001460,0.000709,0.000667,0.000628,0.000831,0.001255,0.001555,0.001691,0.000698,0.001396,0.000911,0.001615,0.001395,0.002126,0.001539,0.001420,0.000894,0.000647,0.001414,0.000523,0.001198,0.000593,0.000553,0.000963,0.001701,0.001329,0.001532,0.001184,0.001310,0.001656,0.001123,0.001829,0.000990,0.001658,0.001479,0.001025,0.000910,0.000728,0.001418,0.000534,0.001028,0.001265,0.001095,0.001033,0.001121,0.001376,0.001060,0.001602,0.001482,0.001207,0.001746,0.001877,0.001239,0.001204,0.001660,0.001369],
    [0.000314,0.000313,0.000209,0.000002,0.001210,0.000003,0.000002,0.000130,0.000589,0.000207,0.000188,0.000081,0.000695,0.000525,0.001148,0.001044,0.001067,0.000772,0.000979,0.000877,0.001117,0.000842,0.001435,0.001041,0.001391,0.000631,0.001093,0.000889,0.001363,0.000862,0.001287,0.000452,0.000732,0.000758,0.000457,0.001678,0.000866,0.000766,0.001290,0.001385,0.001092,0.001043,0.001513,0.001148,0.000914,0.000951,0.001628,0.001155,0.001031,0.001509,0.001314,0.000822,0.001085,0.001505,0.000927,0.000853,0.001209,0.001111,0.000750,0.000818,0.001333,0.000662,0.001324,0.000997,0.001224,0.001878,0.001223,0.000564,0.000993,0.001627,0.001616],
    [0.000507,0.000314,0.000210,0.000002,0.001090,0.000003,0.000005,0.000257,0.000520,0.000380,0.000360,0.000071,0.000717,0.000972,0.000850,0.001043,0.000526,0.000708,0.001463,0.000899,0.000991,0.001104,0.000971,0.000828,0.000800,0.000818,0.000790,0.000730,0.000771,0.000968,0.000709,0.000553,0.000711,0.000880,0.000391,0.001093,0.000601,0.000495,0.001113,0.001856,0.000757,0.001648,0.001002,0.000829,0.000519,0.000740,0.000883,0.001276,0.000435,0.000608,0.001181,0.000421,0.000831,0.000765,0.000949,0.000797,0.001105,0.000790,0.000760,0.001023,0.001024,0.000416,0.001397,0.000799,0.000763,0.001219,0.000693,0.000511,0.000908,0.001103,0.000788],
    [0.000038,0.000146,0.000091,0.000001,0.000204,0.000001,0.000001,0.000030,0.000051,0.000055,0.000063,0.000007,0.000018,0.000322,0.000051,0.000207,0.000101,0.000024,0.000359,0.000513,0.000783,0.000277,0.000230,0.000668,0.001079,0.000447,0.000594,0.000463,0.000981,0.000034,0.000927,0.001558,0.000574,0.001557,0.000982,0.001109,0.000532,0.000086,0.000076,0.000901,0.000369,0.000179,0.000234,0.001312,0.000541,0.000671,0.000703,0.000732,0.000429,0.000680,0.000809,0.000093,0.000734,0.000400,0.000252,0.000488,0.001594,0.001645,0.000454,0.000315,0.000826,0.000134,0.000072,0.000371,0.000340,0.000857,0.000231,0.000176,0.002118,0.001610,0.001977],
    [0.000277,0.000332,0.000221,0.000002,0.001186,0.000003,0.000003,0.000230,0.000384,0.000107,0.000100,0.000053,0.000572,0.001033,0.000996,0.001051,0.000543,0.000685,0.001310,0.001065,0.001622,0.000954,0.000916,0.000934,0.000531,0.000821,0.001528,0.001119,0.000857,0.000893,0.001111,0.001243,0.000783,0.001035,0.000663,0.001131,0.001569,0.000673,0.001584,0.001881,0.000942,0.001134,0.001503,0.001724,0.001369,0.000628,0.000791,0.000964,0.000991,0.001195,0.001652,0.000409,0.000891,0.001306,0.000956,0.001260,0.001347,0.000880,0.000950,0.000875,0.001201,0.001380,0.001127,0.001147,0.001323,0.001573,0.001475,0.000905,0.000936,0.001455,0.001146],
    [0.001570,0.001249,0.000965,0.000015,0.010343,0.000022,0.000014,0.000626,0.001186,0.000791,0.000659,0.000163,0.003471,0.000680,0.000732,0.001563,0.000931,0.001056,0.001087,0.001518,0.001005,0.001477,0.000961,0.000910,0.001295,0.000630,0.001104,0.000998,0.000983,0.000719,0.000895,0.000794,0.000902,0.000935,0.000762,0.000939,0.000772,0.000707,0.000659,0.001878,0.000939,0.000969,0.000946,0.001013,0.001329,0.000653,0.000903,0.000685,0.000737,0.001155,0.000900,0.000717,0.000594,0.001437,0.000848,0.000653,0.000790,0.000936,0.000720,0.000830,0.000711,0.000541,0.001796,0.000919,0.001589,0.000915,0.000918,0.000533,0.000769,0.000749,0.001043],
    [0.001399,0.000978,0.000766,0.000011,0.022378,0.000022,0.000014,0.000585,0.001472,0.000667,0.000546,0.000202,0.003138,0.000864,0.000723,0.000749,0.000765,0.001182,0.001173,0.000655,0.001014,0.001199,0.000948,0.000776,0.001212,0.001110,0.001091,0.001128,0.002133,0.000995,0.000781,0.000857,0.001273,0.000756,0.000614,0.001364,0.001027,0.001015,0.002015,0.001291,0.000683,0.001219,0.000850,0.001471,0.001261,0.001073,0.000975,0.000971,0.001074,0.001422,0.000797,0.000846,0.001068,0.001034,0.001366,0.000931,0.001623,0.001358,0.001264,0.001220,0.000815,0.000993,0.001454,0.001010,0.001218,0.001939,0.001606,0.001561,0.001450,0.001365,0.001246],
    [0.000193,0.000104,0.000062,0.000000,0.000281,0.000001,0.000001,0.000079,0.000167,0.000048,0.000048,0.000023,0.000422,0.000221,0.000368,0.000283,0.000337,0.000551,0.000236,0.000270,0.000326,0.000259,0.000522,0.000233,0.000475,0.000177,0.000346,0.000825,0.000437,0.000232,0.000335,0.000359,0.000172,0.000561,0.000249,0.000863,0.000268,0.000198,0.000298,0.000782,0.000675,0.000986,0.000507,0.000288,0.000297,0.000512,0.000636,0.001563,0.000504,0.000609,0.000708,0.000266,0.000604,0.000399,0.000282,0.000816,0.000644,0.000341,0.001040,0.000406,0.000977,0.000986,0.001135,0.000439,0.000822,0.000486,0.000720,0.000486,0.000389,0.000707,0.002016],
    [0.000167,0.000161,0.000102,0.000001,0.000639,0.000001,0.000002,0.000120,0.000226,0.000069,0.000064,0.000031,0.000376,0.000216,0.000763,0.000913,0.001245,0.000597,0.000664,0.001021,0.001036,0.000999,0.000698,0.000787,0.000672,0.000893,0.001165,0.000674,0.001016,0.000627,0.000601,0.000563,0.000800,0.000530,0.000889,0.000855,0.000523,0.000410,0.001324,0.001826,0.000982,0.000616,0.001216,0.001214,0.000638,0.000967,0.000536,0.000791,0.000404,0.001000,0.001286,0.000465,0.000564,0.000787,0.000589,0.000895,0.001560,0.000676,0.000839,0.000727,0.000749,0.000713,0.001167,0.000948,0.001222,0.001490,0.000883,0.000667,0.000764,0.001379,0.001104],
    [0.000218,0.000283,0.000180,0.000001,0.002395,0.000002,0.000002,0.000116,0.000410,0.000072,0.000067,0.000056,0.000494,0.000290,0.000967,0.001710,0.000857,0.001297,0.000780,0.001031,0.000659,0.001169,0.000568,0.000959,0.000910,0.000357,0.000723,0.001290,0.000654,0.000667,0.000961,0.001163,0.001008,0.000486,0.000947,0.000939,0.001054,0.000618,0.001396,0.001275,0.000979,0.001091,0.001188,0.000700,0.000988,0.000689,0.000976,0.000745,0.000601,0.000901,0.000994,0.001134,0.000671,0.001365,0.001305,0.000678,0.001034,0.000895,0.000875,0.000944,0.000668,0.000537,0.002001,0.001316,0.001681,0.001246,0.000865,0.000784,0.000792,0.000844,0.001377],
    [0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000],
    [0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000],
    [0.000967,0.000314,0.000196,0.000002,0.000282,0.000002,0.000002,0.000091,0.001235,0.000475,0.000420,0.000170,0.001849,0.000719,0.000845,0.000157,0.000608,0.000049,0.000900,0.001199,0.000600,0.001116,0.000406,0.000837,0.000436,0.000831,0.000923,0.000114,0.001507,0.001322,0.001041,0.000323,0.000688,0.000550,0.000649,0.000937,0.000490,0.000575,0.000898,0.000963,0.000999,0.000258,0.000688,0.000527,0.001155,0.000503,0.001262,0.000412,0.000303,0.001126,0.001026,0.000929,0.000825,0.000133,0.000741,0.001513,0.002644,0.001060,0.000707,0.000328,0.000591,0.000509,0.000897,0.000246,0.000853,0.000283,0.000099,0.000335,0.000799,0.000444,0.000932],
    [0.000259,0.000209,0.000130,0.000001,0.001002,0.000001,0.000002,0.000145,0.000339,0.000125,0.000122,0.000047,0.000424,0.000525,0.000745,0.000748,0.000558,0.000572,0.001235,0.000724,0.000985,0.001341,0.000529,0.000475,0.000778,0.000443,0.000716,0.000810,0.000895,0.000888,0.000811,0.000767,0.000522,0.001325,0.000521,0.000653,0.000822,0.000424,0.001131,0.000667,0.000437,0.000910,0.000710,0.000896,0.001160,0.001142,0.000586,0.000638,0.000580,0.001003,0.001041,0.000795,0.000500,0.000903,0.000807,0.000747,0.000598,0.000889,0.000848,0.001293,0.000604,0.001125,0.001616,0.000952,0.000887,0.001261,0.001287,0.000876,0.000786,0.001235,0.001488],
    [0.000280,0.000300,0.000193,0.000001,0.001672,0.000002,0.000003,0.000183,0.000633,0.000123,0.000090,0.000087,0.000731,0.000451,0.000780,0.001220,0.000665,0.000686,0.001236,0.001403,0.001523,0.000872,0.001234,0.001407,0.000590,0.000793,0.001315,0.000800,0.000934,0.000602,0.000610,0.001103,0.000829,0.000831,0.000437,0.001035,0.000606,0.000590,0.000791,0.001277,0.000655,0.000989,0.000881,0.000886,0.001176,0.000740,0.000944,0.000992,0.000863,0.001206,0.000826,0.000597,0.000802,0.001052,0.000665,0.000748,0.001135,0.000997,0.001189,0.000989,0.000591,0.000557,0.000883,0.000713,0.001110,0.001216,0.001087,0.000813,0.000854,0.000886,0.001555]
])
    prob = np.genfromtxt('probabilities.csv')

    def pX_Wi(self, sample):
        x = []
        sigmas = self.sigma
        means = self.mean
        for k in range(28):
            prob = 1
            for i in range(71):
                s = sigmas[k][i]
                if s == 0 :
                    u =0
                else:
                    u = math.exp(-(math.pow((sample[i] - means[k][i]), 2) / (2 * s)))
                    p = (1 / math.sqrt(2 * math.pi * s)) * u
                prob *= p
            x.append(prob)
        return x

    def PWi_X(self, sample):
        Posterior = []
        t = self.pX_Wi(sample)
        for i in range(28):
            post = t[i] * self.prob[i]
            Posterior.append((float(post), i + 1))
        return max(Posterior)

    def call_(self, test):
        posteriors = []
        for i in range(len(test)):
            p = self.PWi_X(test[i])
            posteriors.append(p[1])
        return posteriors

    def accuracy(self, test_data):
        observed_classes = self.call_(test_data)
        correct = 0
        for i in range(len(test_data)):
            c = test_data[i][-1]
            if c == observed_classes[i]:
                correct += 1
        conv = confusion_matrix(observed_classes, test_data[:, [-1]])
        df = pd.DataFrame(conv)
        print(df)
        print(((correct / float(len(test_data))) * 100.0))


if __name__ == '__main__':
    B = Bayesian()
    train_features = np.genfromtxt('Train_images_features.csv', delimiter=',')
    test_features = np.genfromtxt('Test_images_features.csv', delimiter=',')
    B.accuracy(test_features)



















print("total accuracy = 72.4536623"+ ' %')