#include"algorithm/Visualization.h"
namespace pop{
static int edgeTable[256]={
    0x0  , 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c,
    0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00,
    0x190, 0x99 , 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c,
    0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90,
    0x230, 0x339, 0x33 , 0x13a, 0x636, 0x73f, 0x435, 0x53c,
    0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30,
    0x3a0, 0x2a9, 0x1a3, 0xaa , 0x7a6, 0x6af, 0x5a5, 0x4ac,
    0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0,
    0x460, 0x569, 0x663, 0x76a, 0x66 , 0x16f, 0x265, 0x36c,
    0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60,
    0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0xff , 0x3f5, 0x2fc,
    0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0,
    0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x55 , 0x15c,
    0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950,
    0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0xcc ,
    0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0,
    0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc,
    0xcc , 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
    0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c,
    0x15c, 0x55 , 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650,
    0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc,
    0x2fc, 0x3f5, 0xff , 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
    0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c,
    0x36c, 0x265, 0x16f, 0x66 , 0x76a, 0x663, 0x569, 0x460,
    0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac,
    0x4ac, 0x5a5, 0x6af, 0x7a6, 0xaa , 0x1a3, 0x2a9, 0x3a0,
    0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c,
    0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x33 , 0x339, 0x230,
    0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c,
    0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x99 , 0x190,
    0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c,
    0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x0   };
static int triTable[256][16] =
{{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
 {0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
 {0, 1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
 {1, 8, 3, 9, 8, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
 {1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
 {0, 8, 3, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
 {9, 2, 10, 0, 2, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
 {2, 8, 3, 2, 10, 8, 10, 9, 8, -1, -1, -1, -1, -1, -1, -1},
 {3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
 {0, 11, 2, 8, 11, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
 {1, 9, 0, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
 {1, 11, 2, 1, 9, 11, 9, 8, 11, -1, -1, -1, -1, -1, -1, -1},
 {3, 10, 1, 11, 10, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
 {0, 10, 1, 0, 8, 10, 8, 11, 10, -1, -1, -1, -1, -1, -1, -1},
 {3, 9, 0, 3, 11, 9, 11, 10, 9, -1, -1, -1, -1, -1, -1, -1},
 {9, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
 {4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
 {4, 3, 0, 7, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
 {0, 1, 9, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
 {4, 1, 9, 4, 7, 1, 7, 3, 1, -1, -1, -1, -1, -1, -1, -1},
 {1, 2, 10, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
 {3, 4, 7, 3, 0, 4, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1},
 {9, 2, 10, 9, 0, 2, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1},
 {2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4, -1, -1, -1, -1},
 {8, 4, 7, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
 {11, 4, 7, 11, 2, 4, 2, 0, 4, -1, -1, -1, -1, -1, -1, -1},
 {9, 0, 1, 8, 4, 7, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1},
 {4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1, -1, -1, -1, -1},
 {3, 10, 1, 3, 11, 10, 7, 8, 4, -1, -1, -1, -1, -1, -1, -1},
 {1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4, -1, -1, -1, -1},
 {4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3, -1, -1, -1, -1},
 {4, 7, 11, 4, 11, 9, 9, 11, 10, -1, -1, -1, -1, -1, -1, -1},
 {9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
 {9, 5, 4, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
 {0, 5, 4, 1, 5, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
 {8, 5, 4, 8, 3, 5, 3, 1, 5, -1, -1, -1, -1, -1, -1, -1},
 {1, 2, 10, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
 {3, 0, 8, 1, 2, 10, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1},
 {5, 2, 10, 5, 4, 2, 4, 0, 2, -1, -1, -1, -1, -1, -1, -1},
 {2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8, -1, -1, -1, -1},
 {9, 5, 4, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
 {0, 11, 2, 0, 8, 11, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1},
 {0, 5, 4, 0, 1, 5, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1},
 {2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5, -1, -1, -1, -1},
 {10, 3, 11, 10, 1, 3, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1},
 {4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10, -1, -1, -1, -1},
 {5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3, -1, -1, -1, -1},
 {5, 4, 8, 5, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1},
 {9, 7, 8, 5, 7, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
 {9, 3, 0, 9, 5, 3, 5, 7, 3, -1, -1, -1, -1, -1, -1, -1},
 {0, 7, 8, 0, 1, 7, 1, 5, 7, -1, -1, -1, -1, -1, -1, -1},
 {1, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
 {9, 7, 8, 9, 5, 7, 10, 1, 2, -1, -1, -1, -1, -1, -1, -1},
 {10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3, -1, -1, -1, -1},
 {8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2, -1, -1, -1, -1},
 {2, 10, 5, 2, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1},
 {7, 9, 5, 7, 8, 9, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1},
 {9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11, -1, -1, -1, -1},
 {2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7, -1, -1, -1, -1},
 {11, 2, 1, 11, 1, 7, 7, 1, 5, -1, -1, -1, -1, -1, -1, -1},
 {9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11, -1, -1, -1, -1},
 {5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0, -1},
 {11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0, -1},
 {11, 10, 5, 7, 11, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
 {10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
 {0, 8, 3, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
 {9, 0, 1, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
 {1, 8, 3, 1, 9, 8, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1},
 {1, 6, 5, 2, 6, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
 {1, 6, 5, 1, 2, 6, 3, 0, 8, -1, -1, -1, -1, -1, -1, -1},
 {9, 6, 5, 9, 0, 6, 0, 2, 6, -1, -1, -1, -1, -1, -1, -1},
 {5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8, -1, -1, -1, -1},
 {2, 3, 11, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
 {11, 0, 8, 11, 2, 0, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1},
 {0, 1, 9, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1},
 {5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11, -1, -1, -1, -1},
 {6, 3, 11, 6, 5, 3, 5, 1, 3, -1, -1, -1, -1, -1, -1, -1},
 {0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6, -1, -1, -1, -1},
 {3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9, -1, -1, -1, -1},
 {6, 5, 9, 6, 9, 11, 11, 9, 8, -1, -1, -1, -1, -1, -1, -1},
 {5, 10, 6, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
 {4, 3, 0, 4, 7, 3, 6, 5, 10, -1, -1, -1, -1, -1, -1, -1},
 {1, 9, 0, 5, 10, 6, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1},
 {10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4, -1, -1, -1, -1},
 {6, 1, 2, 6, 5, 1, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1},
 {1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7, -1, -1, -1, -1},
 {8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6, -1, -1, -1, -1},
 {7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9, -1},
 {3, 11, 2, 7, 8, 4, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1},
 {5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11, -1, -1, -1, -1},
 {0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1},
 {9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6, -1},
 {8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6, -1, -1, -1, -1},
 {5, 1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11, -1},
 {0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7, -1},
 {6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9, -1, -1, -1, -1},
 {10, 4, 9, 6, 4, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
 {4, 10, 6, 4, 9, 10, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1},
 {10, 0, 1, 10, 6, 0, 6, 4, 0, -1, -1, -1, -1, -1, -1, -1},
 {8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10, -1, -1, -1, -1},
 {1, 4, 9, 1, 2, 4, 2, 6, 4, -1, -1, -1, -1, -1, -1, -1},
 {3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4, -1, -1, -1, -1},
 {0, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
 {8, 3, 2, 8, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1},
 {10, 4, 9, 10, 6, 4, 11, 2, 3, -1, -1, -1, -1, -1, -1, -1},
 {0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6, -1, -1, -1, -1},
 {3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10, -1, -1, -1, -1},
 {6, 4, 1, 6, 1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1, -1},
 {9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3, -1, -1, -1, -1},
 {8, 11, 1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1, -1},
 {3, 11, 6, 3, 6, 0, 0, 6, 4, -1, -1, -1, -1, -1, -1, -1},
 {6, 4, 8, 11, 6, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
 {7, 10, 6, 7, 8, 10, 8, 9, 10, -1, -1, -1, -1, -1, -1, -1},
 {0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10, -1, -1, -1, -1},
 {10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0, -1, -1, -1, -1},
 {10, 6, 7, 10, 7, 1, 1, 7, 3, -1, -1, -1, -1, -1, -1, -1},
 {1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7, -1, -1, -1, -1},
 {2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9, -1},
 {7, 8, 0, 7, 0, 6, 6, 0, 2, -1, -1, -1, -1, -1, -1, -1},
 {7, 3, 2, 6, 7, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
 {2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7, -1, -1, -1, -1},
 {2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7, -1},
 {1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11, -1},
 {11, 2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1, -1, -1, -1, -1},
 {8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6, -1},
 {0, 9, 1, 11, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
 {7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0, -1, -1, -1, -1},
 {7, 11, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
 {7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
 {3, 0, 8, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
 {0, 1, 9, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
 {8, 1, 9, 8, 3, 1, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1},
 {10, 1, 2, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
 {1, 2, 10, 3, 0, 8, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1},
 {2, 9, 0, 2, 10, 9, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1},
 {6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8, -1, -1, -1, -1},
 {7, 2, 3, 6, 2, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
 {7, 0, 8, 7, 6, 0, 6, 2, 0, -1, -1, -1, -1, -1, -1, -1},
 {2, 7, 6, 2, 3, 7, 0, 1, 9, -1, -1, -1, -1, -1, -1, -1},
 {1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6, -1, -1, -1, -1},
 {10, 7, 6, 10, 1, 7, 1, 3, 7, -1, -1, -1, -1, -1, -1, -1},
 {10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8, -1, -1, -1, -1},
 {0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7, -1, -1, -1, -1},
 {7, 6, 10, 7, 10, 8, 8, 10, 9, -1, -1, -1, -1, -1, -1, -1},
 {6, 8, 4, 11, 8, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
 {3, 6, 11, 3, 0, 6, 0, 4, 6, -1, -1, -1, -1, -1, -1, -1},
 {8, 6, 11, 8, 4, 6, 9, 0, 1, -1, -1, -1, -1, -1, -1, -1},
 {9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6, -1, -1, -1, -1},
 {6, 8, 4, 6, 11, 8, 2, 10, 1, -1, -1, -1, -1, -1, -1, -1},
 {1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4, 6, -1, -1, -1, -1},
 {4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9, -1, -1, -1, -1},
 {10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3, -1},
 {8, 2, 3, 8, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1},
 {0, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
 {1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8, -1, -1, -1, -1},
 {1, 9, 4, 1, 4, 2, 2, 4, 6, -1, -1, -1, -1, -1, -1, -1},
 {8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1, -1, -1, -1, -1},
 {10, 1, 0, 10, 0, 6, 6, 0, 4, -1, -1, -1, -1, -1, -1, -1},
 {4, 6, 3, 4, 3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3, -1},
 {10, 9, 4, 6, 10, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
 {4, 9, 5, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
 {0, 8, 3, 4, 9, 5, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1},
 {5, 0, 1, 5, 4, 0, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1},
 {11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5, -1, -1, -1, -1},
 {9, 5, 4, 10, 1, 2, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1},
 {6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5, -1, -1, -1, -1},
 {7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2, -1, -1, -1, -1},
 {3, 4, 8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6, -1},
 {7, 2, 3, 7, 6, 2, 5, 4, 9, -1, -1, -1, -1, -1, -1, -1},
 {9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7, -1, -1, -1, -1},
 {3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0, -1, -1, -1, -1},
 {6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8, -1},
 {9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7, -1, -1, -1, -1},
 {1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4, -1},
 {4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10, -1},
 {7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10, -1, -1, -1, -1},
 {6, 9, 5, 6, 11, 9, 11, 8, 9, -1, -1, -1, -1, -1, -1, -1},
 {3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5, -1, -1, -1, -1},
 {0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11, -1, -1, -1, -1},
 {6, 11, 3, 6, 3, 5, 5, 3, 1, -1, -1, -1, -1, -1, -1, -1},
 {1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6, -1, -1, -1, -1},
 {0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1, 2, 10, -1},
 {11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5, -1},
 {6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3, -1, -1, -1, -1},
 {5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2, -1, -1, -1, -1},
 {9, 5, 6, 9, 6, 0, 0, 6, 2, -1, -1, -1, -1, -1, -1, -1},
 {1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8, -1},
 {1, 5, 6, 2, 1, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
 {1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6, -1},
 {10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0, -1, -1, -1, -1},
 {0, 3, 8, 5, 6, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
 {10, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
 {11, 5, 10, 7, 5, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
 {11, 5, 10, 11, 7, 5, 8, 3, 0, -1, -1, -1, -1, -1, -1, -1},
 {5, 11, 7, 5, 10, 11, 1, 9, 0, -1, -1, -1, -1, -1, -1, -1},
 {10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1, -1, -1, -1, -1},
 {11, 1, 2, 11, 7, 1, 7, 5, 1, -1, -1, -1, -1, -1, -1, -1},
 {0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11, -1, -1, -1, -1},
 {9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7, -1, -1, -1, -1},
 {7, 5, 2, 7, 2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2, -1},
 {2, 5, 10, 2, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1},
 {8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5, -1, -1, -1, -1},
 {9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2, -1, -1, -1, -1},
 {9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2, -1},
 {1, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
 {0, 8, 7, 0, 7, 1, 1, 7, 5, -1, -1, -1, -1, -1, -1, -1},
 {9, 0, 3, 9, 3, 5, 5, 3, 7, -1, -1, -1, -1, -1, -1, -1},
 {9, 8, 7, 5, 9, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
 {5, 8, 4, 5, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1},
 {5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0, -1, -1, -1, -1},
 {0, 1, 9, 8, 4, 10, 8, 10, 11, 10, 4, 5, -1, -1, -1, -1},
 {10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4, -1},
 {2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8, -1, -1, -1, -1},
 {0, 4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11, -1},
 {0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5, -1},
 {9, 4, 5, 2, 11, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
 {2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4, -1, -1, -1, -1},
 {5, 10, 2, 5, 2, 4, 4, 2, 0, -1, -1, -1, -1, -1, -1, -1},
 {3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9, -1},
 {5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2, -1, -1, -1, -1},
 {8, 4, 5, 8, 5, 3, 3, 5, 1, -1, -1, -1, -1, -1, -1, -1},
 {0, 4, 5, 1, 0, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
 {8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5, -1, -1, -1, -1},
 {9, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
 {4, 11, 7, 4, 9, 11, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1},
 {0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11, -1, -1, -1, -1},
 {1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11, -1, -1, -1, -1},
 {3, 1, 4, 3, 4, 8, 1, 10, 4, 7, 4, 11, 10, 11, 4, -1},
 {4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2, -1, -1, -1, -1},
 {9, 7, 4, 9, 11, 7, 9, 1, 11, 2, 11, 1, 0, 8, 3, -1},
 {11, 7, 4, 11, 4, 2, 2, 4, 0, -1, -1, -1, -1, -1, -1, -1},
 {11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4, -1, -1, -1, -1},
 {2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9, -1, -1, -1, -1},
 {9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7, -1},
 {3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10, -1},
 {1, 10, 2, 8, 7, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
 {4, 9, 1, 4, 1, 7, 7, 1, 3, -1, -1, -1, -1, -1, -1, -1},
 {4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1, -1, -1, -1, -1},
 {4, 0, 3, 7, 4, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
 {4, 8, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
 {9, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
 {3, 0, 9, 3, 9, 11, 11, 9, 10, -1, -1, -1, -1, -1, -1, -1},
 {0, 1, 10, 0, 10, 8, 8, 10, 11, -1, -1, -1, -1, -1, -1, -1},
 {3, 1, 10, 11, 3, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
 {1, 2, 11, 1, 11, 9, 9, 11, 8, -1, -1, -1, -1, -1, -1, -1},
 {3, 0, 9, 3, 9, 11, 1, 2, 9, 2, 11, 9, -1, -1, -1, -1},
 {0, 2, 11, 8, 0, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
 {3, 2, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
 {2, 3, 8, 2, 8, 10, 10, 8, 9, -1, -1, -1, -1, -1, -1, -1},
 {9, 10, 2, 0, 9, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
 {2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8, -1, -1, -1, -1},
 {1, 10, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
 {1, 3, 8, 9, 1, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
 {0, 9, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
 {0, 3, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
 {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}};



Visualization::vertex Visualization::interpolate(Visualization::vertex p1, Visualization::vertex p2, F32 p1value, F32 p2value , F32 iso ) {


    vertex p;
    F32 diff;
    diff = (iso - p1value) / (p2value - p1value);


    //    if(p2value>p1value)
    //        diff = (iso - p1value) / (p2value - p1value);
    //    else
    //        diff = (iso - p2value) / (p1value - p2value);
    p.x = p1.x + diff * (p2.x - p1.x);
    p.y = p1.y + diff * (p2.y - p1.y);
    p.z = p1.z + diff * (p2.z - p1.z);

    p.normal_x = p1.normal_x + diff * (p2.normal_x - p1.normal_x);
    p.normal_y = p1.normal_y + diff * (p2.normal_y - p1.normal_y);
    p.normal_z = p1.normal_z + diff * (p2.normal_z - p1.normal_z);
    F32 sum = std::sqrt(p.normal_x*p.normal_x+ p.normal_y*p.normal_y+p.normal_z*p.normal_z);
    if(sum==0){
        F32 sum = std::sqrt(p1.normal_x*p1.normal_x+ p1.normal_y*p1.normal_y+p1.normal_z*p1.normal_z);


        p.normal_x=p1.normal_x/sum;
        p.normal_y=p1.normal_x/sum;
        p.normal_z=p1.normal_x/sum;
    }
    else{
        p.normal_x/=sum;
        p.normal_y/=sum;
        p.normal_z/=sum;
    }

    return p;
}

void Visualization::processCube(cubeF cube,std::vector<vertex>& vertexList,F32 isolevel ,bool diff) {

    //if(value<isolevel)
    {
        int cubeindex = 0;
        if(cube.val[0] > isolevel) cubeindex |= 1;
        if(cube.val[1] > isolevel) cubeindex |= 2;
        if(cube.val[2] > isolevel) cubeindex |= 4;
        if(cube.val[3] > isolevel) cubeindex |= 8;
        if(cube.val[4] > isolevel) cubeindex |= 16;
        if(cube.val[5] > isolevel) cubeindex |= 32;
        if(cube.val[6] > isolevel) cubeindex |= 64;
        if(cube.val[7] > isolevel) cubeindex |= 128;

        // Cube is entirely in/out of the surface
        if(edgeTable[cubeindex] == 0 || edgeTable[cubeindex] == 255)
            return;

        vertex vertlist[12];
        // Find the vertices where the surface intersects the cube
        if(diff==false){
            if(edgeTable[cubeindex] & 1)
                vertlist[0] = interpolate(cube.p[0],cube.p[1]);
            if(edgeTable[cubeindex] & 2)
                vertlist[1] = interpolate(cube.p[1],cube.p[2]);
            if(edgeTable[cubeindex] & 4)
                vertlist[2] = interpolate(cube.p[2],cube.p[3]);
            if(edgeTable[cubeindex] & 8)
                vertlist[3] = interpolate(cube.p[3],cube.p[0]);
            if(edgeTable[cubeindex] & 16)
                vertlist[4] = interpolate(cube.p[4],cube.p[5]);
            if(edgeTable[cubeindex] & 32)
                vertlist[5] = interpolate(cube.p[5],cube.p[6]);
            if(edgeTable[cubeindex] & 64)
                vertlist[6] = interpolate(cube.p[6],cube.p[7]);
            if(edgeTable[cubeindex] & 128)
                vertlist[7] = interpolate(cube.p[7],cube.p[4]);
            if(edgeTable[cubeindex] & 256)
                vertlist[8] = interpolate(cube.p[0],cube.p[4]);
            if(edgeTable[cubeindex] & 512)
                vertlist[9] = interpolate(cube.p[1],cube.p[5]);
            if(edgeTable[cubeindex] & 1024)
                vertlist[10] = interpolate(cube.p[2],cube.p[6]);
            if(edgeTable[cubeindex] & 2048)
                vertlist[11] = interpolate(cube.p[3],cube.p[7]);
        }
        else{
            if(edgeTable[cubeindex] & 1)
                vertlist[0] = interpolate(cube.p[0],cube.p[1],cube.val[0],cube.val[1],isolevel);
            if(edgeTable[cubeindex] & 2)
                vertlist[1] = interpolate(cube.p[1],cube.p[2],cube.val[1],cube.val[2],isolevel);
            if(edgeTable[cubeindex] & 4)
                vertlist[2] = interpolate(cube.p[2],cube.p[3],cube.val[2],cube.val[3],isolevel);
            if(edgeTable[cubeindex] & 8)
                vertlist[3] = interpolate(cube.p[3],cube.p[0],cube.val[3],cube.val[0],isolevel);
            if(edgeTable[cubeindex] & 16)
                vertlist[4] = interpolate(cube.p[4],cube.p[5],cube.val[4],cube.val[5],isolevel);
            if(edgeTable[cubeindex] & 32)
                vertlist[5] = interpolate(cube.p[5],cube.p[6],cube.val[5],cube.val[6],isolevel);
            if(edgeTable[cubeindex] & 64)
                vertlist[6] = interpolate(cube.p[6],cube.p[7],cube.val[6],cube.val[7],isolevel);
            if(edgeTable[cubeindex] & 128)
                vertlist[7] = interpolate(cube.p[7],cube.p[4],cube.val[7],cube.val[4],isolevel);
            if(edgeTable[cubeindex] & 256)
                vertlist[8] = interpolate(cube.p[0],cube.p[4],cube.val[0],cube.val[4],isolevel);
            if(edgeTable[cubeindex] & 512)
                vertlist[9] = interpolate(cube.p[1],cube.p[5],cube.val[1],cube.val[5],isolevel);
            if(edgeTable[cubeindex] & 1024)
                vertlist[10] = interpolate(cube.p[2],cube.p[6],cube.val[2],cube.val[6],isolevel);
            if(edgeTable[cubeindex] & 2048)
                vertlist[11] = interpolate(cube.p[3],cube.p[7],cube.val[3],cube.val[7],isolevel);
        }

        for(int i = 0; triTable[cubeindex][i] != -1; i++) {
            vertexList.push_back(vertlist[triTable[cubeindex][i]]);
        }
    }
}



void Visualization::processCubeIso(Visualization::Cube cube, std::vector<std::pair<vertex,RGBUI8 > >& vertexList,RGBUI8 value,unsigned char isolevel) {
    //if(value<isolevel)
    {
        int cubeindex = 0;
        if(cube.val[0] > isolevel) cubeindex |= 1;
        if(cube.val[1] > isolevel) cubeindex |= 2;
        if(cube.val[2] > isolevel) cubeindex |= 4;
        if(cube.val[3] > isolevel) cubeindex |= 8;
        if(cube.val[4] > isolevel) cubeindex |= 16;
        if(cube.val[5] > isolevel) cubeindex |= 32;
        if(cube.val[6] > isolevel) cubeindex |= 64;
        if(cube.val[7] > isolevel) cubeindex |= 128;

        // Cube is entirely in/out of the surface
        if(edgeTable[cubeindex] == 0 || edgeTable[cubeindex] == 255)
            return;

        vertex vertlist[12];
        // Find the vertices where the surface intersects the cube
        if(edgeTable[cubeindex] & 1)
            vertlist[0] = interpolate(cube.p[0],cube.p[1]);
        if(edgeTable[cubeindex] & 2)
            vertlist[1] = interpolate(cube.p[1],cube.p[2]);
        if(edgeTable[cubeindex] & 4)
            vertlist[2] = interpolate(cube.p[2],cube.p[3]);
        if(edgeTable[cubeindex] & 8)
            vertlist[3] = interpolate(cube.p[3],cube.p[0]);
        if(edgeTable[cubeindex] & 16)
            vertlist[4] = interpolate(cube.p[4],cube.p[5]);
        if(edgeTable[cubeindex] & 32)
            vertlist[5] = interpolate(cube.p[5],cube.p[6]);
        if(edgeTable[cubeindex] & 64)
            vertlist[6] = interpolate(cube.p[6],cube.p[7]);
        if(edgeTable[cubeindex] & 128)
            vertlist[7] = interpolate(cube.p[7],cube.p[4]);
        if(edgeTable[cubeindex] & 256)
            vertlist[8] = interpolate(cube.p[0],cube.p[4]);
        if(edgeTable[cubeindex] & 512)
            vertlist[9] = interpolate(cube.p[1],cube.p[5]);
        if(edgeTable[cubeindex] & 1024)
            vertlist[10] = interpolate(cube.p[2],cube.p[6]);
        if(edgeTable[cubeindex] & 2048)
            vertlist[11] = interpolate(cube.p[3],cube.p[7]);

        for(int i = 0; triTable[cubeindex][i] != -1; i++) {
            vertexList.push_back(std::make_pair(vertlist[triTable[cubeindex][i]] ,value  ));
        }
    }
}


void Visualization::processCube(Visualization::Cube cube, std::vector<std::pair<vertex,RGBUI8 > >& vertexList,RGBUI8 value,bool diff) {
    //if(value<isolevel)
    F32 isolevel =0.5;
    {
        int cubeindex = 0;
        if(cube.val[0] > isolevel) cubeindex |= 1;
        if(cube.val[1] > isolevel) cubeindex |= 2;
        if(cube.val[2] > isolevel) cubeindex |= 4;
        if(cube.val[3] > isolevel) cubeindex |= 8;
        if(cube.val[4] > isolevel) cubeindex |= 16;
        if(cube.val[5] > isolevel) cubeindex |= 32;
        if(cube.val[6] > isolevel) cubeindex |= 64;
        if(cube.val[7] > isolevel) cubeindex |= 128;

        // Cube is entirely in/out of the surface
        if(edgeTable[cubeindex] == 0 || edgeTable[cubeindex] == 255)
            return;

        vertex vertlist[12];
        // Find the vertices where the surface intersects the cube
        if(diff==false){
            if(edgeTable[cubeindex] & 1)
                vertlist[0] = interpolate(cube.p[0],cube.p[1]);
            if(edgeTable[cubeindex] & 2)
                vertlist[1] = interpolate(cube.p[1],cube.p[2]);
            if(edgeTable[cubeindex] & 4)
                vertlist[2] = interpolate(cube.p[2],cube.p[3]);
            if(edgeTable[cubeindex] & 8)
                vertlist[3] = interpolate(cube.p[3],cube.p[0]);
            if(edgeTable[cubeindex] & 16)
                vertlist[4] = interpolate(cube.p[4],cube.p[5]);
            if(edgeTable[cubeindex] & 32)
                vertlist[5] = interpolate(cube.p[5],cube.p[6]);
            if(edgeTable[cubeindex] & 64)
                vertlist[6] = interpolate(cube.p[6],cube.p[7]);
            if(edgeTable[cubeindex] & 128)
                vertlist[7] = interpolate(cube.p[7],cube.p[4]);
            if(edgeTable[cubeindex] & 256)
                vertlist[8] = interpolate(cube.p[0],cube.p[4]);
            if(edgeTable[cubeindex] & 512)
                vertlist[9] = interpolate(cube.p[1],cube.p[5]);
            if(edgeTable[cubeindex] & 1024)
                vertlist[10] = interpolate(cube.p[2],cube.p[6]);
            if(edgeTable[cubeindex] & 2048)
                vertlist[11] = interpolate(cube.p[3],cube.p[7]);
        }
        else{
            if(edgeTable[cubeindex] & 1)
                vertlist[0] = interpolate(cube.p[0],cube.p[1],cube.val[0],cube.val[1]);
            if(edgeTable[cubeindex] & 2)
                vertlist[1] = interpolate(cube.p[1],cube.p[2],cube.val[1],cube.val[2]);
            if(edgeTable[cubeindex] & 4)
                vertlist[2] = interpolate(cube.p[2],cube.p[3],cube.val[2],cube.val[3]);
            if(edgeTable[cubeindex] & 8)
                vertlist[3] = interpolate(cube.p[3],cube.p[0],cube.val[3],cube.val[0]);
            if(edgeTable[cubeindex] & 16)
                vertlist[4] = interpolate(cube.p[4],cube.p[5],cube.val[4],cube.val[5]);
            if(edgeTable[cubeindex] & 32)
                vertlist[5] = interpolate(cube.p[5],cube.p[6],cube.val[5],cube.val[6]);
            if(edgeTable[cubeindex] & 64)
                vertlist[6] = interpolate(cube.p[6],cube.p[7],cube.val[6],cube.val[7]);
            if(edgeTable[cubeindex] & 128)
                vertlist[7] = interpolate(cube.p[7],cube.p[4],cube.val[7],cube.val[4]);
            if(edgeTable[cubeindex] & 256)
                vertlist[8] = interpolate(cube.p[0],cube.p[4],cube.val[0],cube.val[4]);
            if(edgeTable[cubeindex] & 512)
                vertlist[9] = interpolate(cube.p[1],cube.p[5],cube.val[1],cube.val[5]);
            if(edgeTable[cubeindex] & 1024)
                vertlist[10] = interpolate(cube.p[2],cube.p[6],cube.val[2],cube.val[6]);
            if(edgeTable[cubeindex] & 2048)
                vertlist[11] = interpolate(cube.p[3],cube.p[7],cube.val[3],cube.val[7]);
        }

        for(int i = 0; triTable[cubeindex][i] != -1; i++) {
            vertexList.push_back(std::make_pair(vertlist[triTable[cubeindex][i]] ,value  ));
        }
    }
}

namespace Private{
F32 affectRGB(RGBUI8 c1,RGBUI8 c2){
    F32 vv = c1.lumi()*1.0-c2.lumi()*1.0;
    return vv;

}
}

std::vector<std::pair<Visualization::vertex,RGBUI8 > > Visualization::runMarchingCubesSurfaceContact(const MatN<3,RGB<UI8 > > &voxel) {

    MatN<3,RGB<UI8 > > voxels(voxel.getDomain()+4);
    MatN<3,RGB< UI8 > >::IteratorEDomain it (voxel.getIteratorEDomain());


    while(it.next())
    {
        VecN<3,int> x  =it.x();
        x=x+2;
        voxels(x)=voxel(it.x());
    }

    std::vector<std::pair<vertex,RGBUI8 > > vertexList;
    int sizeX =  voxels.getDomain()(0);
    int sizeY=voxels.getDomain()(1);
    int sizeZ=voxels.getDomain()(2);
    int stepX=1;
    int stepY=1;
    int stepZ=1;


    // Run the processCube function on every cube in the grid
    for(pop::F32 x = stepX; x < sizeX-2*stepX; x += stepX) {
        for(pop::F32 y = stepY; y < sizeY-2*stepY; y += stepY) {
            for(pop::F32 z = stepZ; z < sizeZ-2*stepZ; z += stepZ) {

                RGBUI8 cc=maximum(
                            voxels(VecN<3,int>(x,y,z)),maximum(
                                voxels(VecN<3,int>(x+stepX,y,z)),maximum(
                                    voxels(VecN<3,int>(x+stepX,y,z+stepZ)),maximum(
                                        voxels(VecN<3,int>(x,y,z+stepZ)),maximum(
                                            voxels(VecN<3,int>(x,y+stepY,z)),maximum(
                                                voxels(VecN<3,int>(x+stepX,y+stepY,z)),maximum(
                                                    voxels(VecN<3,int>(x+stepX,y+stepY,z+stepZ)),
                                                    voxels(VecN<3,int>(x,y+stepY,z+stepZ)))))))));
                RGBUI8 ccmin=minimum(
                            voxels(VecN<3,int>(x,y,z)),minimum(
                                voxels(VecN<3,int>(x+stepX,y,z)),minimum(
                                    voxels(VecN<3,int>(x+stepX,y,z+stepZ)),minimum(
                                        voxels(VecN<3,int>(x,y,z+stepZ)),minimum(
                                            voxels(VecN<3,int>(x,y+stepY,z)),minimum(
                                                voxels(VecN<3,int>(x+stepX,y+stepY,z)),minimum(
                                                    voxels(VecN<3,int>(x+stepX,y+stepY,z+stepZ)),
                                                    voxels(VecN<3,int>(x,y+stepY,z+stepZ)))))))));

                Visualization::Cube c = {{
                                             {x,y,z,
                                              Private::affectRGB(voxels(VecN<3,int>(x+stepX,y,z)),voxels(VecN<3,int>(x-stepX,y,z))) / -stepX,
                                              Private::affectRGB(voxels(VecN<3,int>(x,y+stepY,z)),voxels(VecN<3,int>(x,y-stepY,z))) / -stepY,
                                              Private::affectRGB(voxels(VecN<3,int>(x,y,z+stepZ)),voxels(VecN<3,int>(x,y,z-stepZ))) / -stepZ
                                             },
                                             {x+stepX,y,z,
                                              Private::affectRGB(voxels(VecN<3,int>(x+2*stepX,y,z)),voxels(VecN<3,int>(x,y,z))) / -stepX,
                                              Private::affectRGB(voxels(VecN<3,int>(x+stepX,y+stepY,z)),voxels(VecN<3,int>(x+stepX,y-stepY,z))) / -stepY,
                                              Private::affectRGB(voxels(VecN<3,int>(x+stepX,y,z+stepZ)),voxels(VecN<3,int>(x+stepX,y,z-stepZ))) / -stepZ
                                             },
                                             {x+stepX,y,z+stepZ,
                                              Private::affectRGB(voxels(VecN<3,int>(x+2*stepX,y,z+stepZ)),voxels(VecN<3,int>(x,y,z+stepZ))) / -stepX,
                                              Private::affectRGB(voxels(VecN<3,int>(x+stepX,y+stepY,z+stepZ)),voxels(VecN<3,int>(x+stepX,y-stepY,z+stepZ))) / -stepY,
                                              Private::affectRGB(voxels(VecN<3,int>(x+stepX,y,z+2*stepZ)),voxels(VecN<3,int>(x+stepX,y,z))) / -stepZ
                                             },
                                             {x,y,z+stepZ,
                                              Private::affectRGB(voxels(VecN<3,int>(x+stepX,y,z+stepZ)),voxels(VecN<3,int>(x-stepX,y,z+stepZ))) / -stepX,
                                              Private::affectRGB(voxels(VecN<3,int>(x,y+stepY,z+stepZ)),voxels(VecN<3,int>(x,y-stepY,z+stepZ))) / -stepY,
                                              Private::affectRGB(voxels(VecN<3,int>(x,y,z+2*stepZ)),voxels(VecN<3,int>(x,y,z))) / -stepZ
                                             },
                                             {x,y+stepY,z,
                                              Private::affectRGB(voxels(VecN<3,int>(x+stepX,y+stepY,z)),voxels(VecN<3,int>(x-stepX,y+stepY,z))) / -stepX,
                                              Private::affectRGB(voxels(VecN<3,int>(x,y+2*stepY,z)),voxels(VecN<3,int>(x,y,z))) / -stepY,
                                              Private::affectRGB(voxels(VecN<3,int>(x,y+stepY,z+stepZ)),voxels(VecN<3,int>(x,y+stepY,z-stepZ))) / -stepZ
                                             },
                                             {x+stepX,y+stepY,z,
                                              Private::affectRGB(voxels(VecN<3,int>(x+2*stepX,y+stepY,z)),voxels(VecN<3,int>(x+stepX,y+stepY,z))) / -stepX,
                                              Private::affectRGB(voxels(VecN<3,int>(x+stepX,y+2*stepY,z)),voxels(VecN<3,int>(x+stepX,y,z))) / -stepY,
                                              Private::affectRGB(voxels(VecN<3,int>(x+stepX,y+stepY,z+stepZ)),voxels(VecN<3,int>(x+stepX,y+stepY,z-stepZ))) / -stepZ
                                             },
                                             {x+stepX,y+stepY,z+stepZ,
                                              Private::affectRGB(voxels(VecN<3,int>(x+2*stepX,y+stepY,z+stepZ)),voxels(VecN<3,int>(x,y+stepY,z+stepZ))) / -stepX,
                                              Private::affectRGB(voxels(VecN<3,int>(x+stepX,y+2*stepY,z+stepZ)),voxels(VecN<3,int>(x+stepX,y,z+stepZ))) / -stepY,
                                              Private::affectRGB(voxels(VecN<3,int>(x+stepX,y+stepY,z+2*stepZ)),voxels(VecN<3,int>(x+stepX,y+stepY,z))) / -stepZ
                                             },
                                             {x,y+stepY,z+stepZ,
                                              Private::affectRGB(voxels(VecN<3,int>(x+stepX,y+stepY,z+stepZ)),voxels(VecN<3,int>(x-stepX,y+stepY,z+stepZ))) / -stepX,
                                              Private::affectRGB(voxels(VecN<3,int>(x,y+2*stepY,z+stepZ)),voxels(VecN<3,int>(x,y,z+stepZ))) / -stepY,
                                              Private::affectRGB(voxels(VecN<3,int>(x,y+stepY,z+2*stepZ)),voxels(VecN<3,int>(x,y+stepY,z))) / -stepZ
                                             }
                                         },{
                                             static_cast<pop::F32>(voxels(VecN<3,int>(x,y,z)).lumi()),
                                             static_cast<pop::F32>(voxels(VecN<3,int>(x+stepX,y,z)).lumi()),
                                             static_cast<pop::F32>(voxels(VecN<3,int>(x+stepX,y,z+stepZ)).lumi()),
                                             static_cast<pop::F32>(voxels(VecN<3,int>(x,y,z+stepZ)).lumi()),
                                             static_cast<pop::F32>(voxels(VecN<3,int>(x,y+stepY,z)).lumi()),
                                             static_cast<pop::F32>(voxels(VecN<3,int>(x+stepX,y+stepY,z)).lumi()),
                                             static_cast<pop::F32>(voxels(VecN<3,int>(x+stepX,y+stepY,z+stepZ)).lumi()),
                                             static_cast<pop::F32>(voxels(VecN<3,int>(x,y+stepY,z+stepZ)).lumi())
                                         }};
                if(ccmin.lumi()>=1)
                    processCubeIso(c, vertexList,cc,ccmin.lumi());

            }
        }
    }

    return vertexList;
}

std::vector<std::pair<Visualization::vertex,RGBUI8 > > Visualization::runMarchingCubes2(const MatN<3,RGB<UI8 > > &voxel) {

    MatN<3,RGB<UI8 > > voxels(voxel.getDomain()+4);
    MatN<3,RGB< UI8 > >::IteratorEDomain it (voxel.getIteratorEDomain());


    while(it.next())
    {
        VecN<3,int> x  =it.x();
        x=x+2;
        voxels(x)=voxel(it.x());
    }

    std::vector<std::pair<vertex,RGBUI8 > > vertexList;
    int sizeX =  voxels.getDomain()(0);
    int sizeY=voxels.getDomain()(1);
    int sizeZ=voxels.getDomain()(2);
    int stepX=1;
    int stepY=1;
    int stepZ=1;


    // Run the processCube function on every cube in the grid
    for(pop::F32 x = stepX; x < sizeX-2*stepX; x += stepX) {
        for(pop::F32 y = stepY; y < sizeY-2*stepY; y += stepY) {
            for(pop::F32 z = stepZ; z < sizeZ-2*stepZ; z += stepZ) {

                RGBUI8 cc=maximum(
                            voxels(VecN<3,int>(x,y,z)),maximum(
                                voxels(VecN<3,int>(x+stepX,y,z)),maximum(
                                    voxels(VecN<3,int>(x+stepX,y,z+stepZ)),maximum(
                                        voxels(VecN<3,int>(x,y,z+stepZ)),maximum(
                                            voxels(VecN<3,int>(x,y+stepY,z)),maximum(
                                                voxels(VecN<3,int>(x+stepX,y+stepY,z)),maximum(
                                                    voxels(VecN<3,int>(x+stepX,y+stepY,z+stepZ)),
                                                    voxels(VecN<3,int>(x,y+stepY,z+stepZ)))))))));
                Visualization::Cube c = {{
                                             {x,y,z,
                                              Private::affectRGB(voxels(VecN<3,int>(x+stepX,y,z)),voxels(VecN<3,int>(x-stepX,y,z))) / -stepX,
                                              Private::affectRGB(voxels(VecN<3,int>(x,y+stepY,z)),voxels(VecN<3,int>(x,y-stepY,z))) / -stepY,
                                              Private::affectRGB(voxels(VecN<3,int>(x,y,z+stepZ)),voxels(VecN<3,int>(x,y,z-stepZ))) / -stepZ
                                             },
                                             {x+stepX,y,z,
                                              Private::affectRGB(voxels(VecN<3,int>(x+2*stepX,y,z)),    voxels(VecN<3,int>(x,y,z))) / -stepX,
                                              Private::affectRGB(voxels(VecN<3,int>(x+stepX,y+stepY,z)),voxels(VecN<3,int>(x+stepX,y-stepY,z))) / -stepY,
                                              Private::affectRGB(voxels(VecN<3,int>(x+stepX,y,z+stepZ)),voxels(VecN<3,int>(x+stepX,y,z-stepZ))) / -stepZ
                                             },
                                             {x+stepX,y,z+stepZ,
                                              Private::affectRGB(voxels(VecN<3,int>(x+2*stepX,y,z+stepZ)),    voxels(VecN<3,int>(x,y,z+stepZ))) / -stepX,
                                              Private::affectRGB(voxels(VecN<3,int>(x+stepX,y+stepY,z+stepZ)),voxels(VecN<3,int>(x+stepX,y-stepY,z+stepZ))) / -stepY,
                                              Private::affectRGB(voxels(VecN<3,int>(x+stepX,y,z+2*stepZ)),    voxels(VecN<3,int>(x+stepX,y,z))) / -stepZ
                                             },
                                             {x,y,z+stepZ,
                                              Private::affectRGB(voxels(VecN<3,int>(x+stepX,y,z+stepZ)),voxels(VecN<3,int>(x-stepX,y,z+stepZ))) / -stepX,
                                              Private::affectRGB(voxels(VecN<3,int>(x,y+stepY,z+stepZ)),voxels(VecN<3,int>(x,y-stepY,z+stepZ))) / -stepY,
                                              Private::affectRGB(voxels(VecN<3,int>(x,y,z+2*stepZ)),    voxels(VecN<3,int>(x,y,z))) / -stepZ
                                             },
                                             {x,y+stepY,z,
                                              Private::affectRGB(voxels(VecN<3,int>(x+stepX,y+stepY,z)),voxels(VecN<3,int>(x-stepX,y+stepY,z))) / -stepX,
                                              Private::affectRGB(voxels(VecN<3,int>(x,y+2*stepY,z)),   voxels(VecN<3,int>(x,y,z))) / -stepY,
                                              Private::affectRGB(voxels(VecN<3,int>(x,y+stepY,z+stepZ)),voxels(VecN<3,int>(x,y+stepY,z-stepZ))) / -stepZ
                                             },
                                             {x+stepX,y+stepY,z,
                                              Private::affectRGB(voxels(VecN<3,int>(x+2*stepX,y+stepY,z)),    voxels(VecN<3,int>(x,y+stepY,z))) / -stepX,
                                              Private::affectRGB(voxels(VecN<3,int>(x+stepX,y+2*stepY,z)),    voxels(VecN<3,int>(x+stepX,y,z))) / -stepY,
                                              Private::affectRGB(voxels(VecN<3,int>(x+stepX,y+stepY,z+stepZ)),voxels(VecN<3,int>(x+stepX,y+stepY,z-stepZ))) / -stepZ
                                             },
                                             {x+stepX,y+stepY,z+stepZ,
                                              Private::affectRGB(voxels(VecN<3,int>(x+2*stepX,y+stepY,z+stepZ)),voxels(VecN<3,int>(x,y+stepY,z+stepZ))) / -stepX,
                                              Private::affectRGB(voxels(VecN<3,int>(x+stepX,y+2*stepY,z+stepZ)),voxels(VecN<3,int>(x+stepX,y,z+stepZ))) / -stepY,
                                              Private::affectRGB(voxels(VecN<3,int>(x+stepX,y+stepY,z+2*stepZ)),voxels(VecN<3,int>(x+stepX,y+stepY,z))) / -stepZ
                                             },
                                             {x,y+stepY,z+stepZ,
                                              Private::affectRGB(voxels(VecN<3,int>(x+stepX,y+stepY,z+stepZ)),voxels(VecN<3,int>(x-stepX,y+stepY,z+stepZ))) / -stepX,
                                              Private::affectRGB(voxels(VecN<3,int>(x,y+2*stepY,z+stepZ)),    voxels(VecN<3,int>(x,y,z+stepZ))) / -stepY,
                                              Private::affectRGB(voxels(VecN<3,int>(x,y+stepY,z+2*stepZ)),    voxels(VecN<3,int>(x,y+stepY,z))) / -stepZ
                                             }
                                         },{
                                             static_cast<pop::F32>(voxels(VecN<3,int>(x,y,z)).lumi()),
                                             static_cast<pop::F32>(voxels(VecN<3,int>(x+stepX,y,z)).lumi()),
                                             static_cast<pop::F32>(voxels(VecN<3,int>(x+stepX,y,z+stepZ)).lumi()),
                                             static_cast<pop::F32>(voxels(VecN<3,int>(x,y,z+stepZ)).lumi()),
                                             static_cast<pop::F32>(voxels(VecN<3,int>(x,y+stepY,z)).lumi()),
                                             static_cast<pop::F32>(voxels(VecN<3,int>(x+stepX,y+stepY,z)).lumi()),
                                             static_cast<pop::F32>(voxels(VecN<3,int>(x+stepX,y+stepY,z+stepZ)).lumi()),
                                             static_cast<pop::F32>(voxels(VecN<3,int>(x,y+stepY,z+stepZ)).lumi())
                                         }};

                //                Visualization::Cube c = {{
                //                              {x,y,z,
                //                               Private::affectRGB(voxels(VecN<3,int>(x+stepX,y,z)),voxels(VecN<3,int>(x,y,z))) / -stepX,
                //                               Private::affectRGB(voxels(VecN<3,int>(x,y+stepY,z)),voxels(VecN<3,int>(x,y,z))) / -stepY,
                //                               Private::affectRGB(voxels(VecN<3,int>(x,y,z+stepZ)),voxels(VecN<3,int>(x,y,z))) / -stepZ
                //                              },
                //                              {x+stepX,y,z,
                //                               Private::affectRGB(voxels(VecN<3,int>(x+2*stepX,y,z)),    voxels(VecN<3,int>(x+stepX,y,z))) / -stepX,
                //                               Private::affectRGB(voxels(VecN<3,int>(x+stepX,y+stepY,z)),voxels(VecN<3,int>(x+stepX,y,z))) / -stepY,
                //                               Private::affectRGB(voxels(VecN<3,int>(x+stepX,y,z+stepZ)),voxels(VecN<3,int>(x+stepX,y,z))) / -stepZ
                //                              },
                //                              {x+stepX,y,z+stepZ,
                //                               Private::affectRGB(voxels(VecN<3,int>(x+2*stepX,y,z+stepZ)),    voxels(VecN<3,int>(x+stepX,y,z+stepZ))) / -stepX,
                //                               Private::affectRGB(voxels(VecN<3,int>(x+stepX,y+stepY,z+stepZ)),voxels(VecN<3,int>(x+stepX,y,z+stepZ))) / -stepY,
                //                               Private::affectRGB(voxels(VecN<3,int>(x+stepX,y,z+2*stepZ)),    voxels(VecN<3,int>(x+stepX,y,z+stepZ))) / -stepZ
                //                              },
                //                              {x,y,z+stepZ,
                //                               Private::affectRGB(voxels(VecN<3,int>(x+stepX,y,z+stepZ)),voxels(VecN<3,int>(x,y,z+stepZ))) / -stepX,
                //                               Private::affectRGB(voxels(VecN<3,int>(x,y+stepY,z+stepZ)),voxels(VecN<3,int>(x,y,z+stepZ))) / -stepY,
                //                               Private::affectRGB(voxels(VecN<3,int>(x,y,z+2*stepZ)),    voxels(VecN<3,int>(x,y,z+stepZ))) / -stepZ
                //                              },
                //                              {x,y+stepY,z,
                //                               Private::affectRGB(voxels(VecN<3,int>(x+stepX,y+stepY,z)),voxels(VecN<3,int>(x,y+stepY,z))) / -stepX,
                //                               Private::affectRGB(voxels(VecN<3,int>(x,y+2*stepY,z)),    voxels(VecN<3,int>(x,y+stepY,z))) / -stepY,
                //                               Private::affectRGB(voxels(VecN<3,int>(x,y+stepY,z+stepZ)),voxels(VecN<3,int>(x,y+stepY,z))) / -stepZ
                //                              },
                //                              {x+stepX,y+stepY,z,
                //                               Private::affectRGB(voxels(VecN<3,int>(x+2*stepX,y+stepY,z)),    voxels(VecN<3,int>(x+stepX,y+stepY,z))) / -stepX,
                //                               Private::affectRGB(voxels(VecN<3,int>(x+stepX,y+2*stepY,z)),    voxels(VecN<3,int>(x+stepX,y+stepY,z))) / -stepY,
                //                               Private::affectRGB(voxels(VecN<3,int>(x+stepX,y+stepY,z+stepZ)),voxels(VecN<3,int>(x+stepX,y+stepY,z))) / -stepZ
                //                              },
                //                              {x+stepX,y+stepY,z+stepZ,
                //                               Private::affectRGB(voxels(VecN<3,int>(x+2*stepX,y+stepY,z+stepZ)),voxels(VecN<3,int>(x+stepX,y+stepY,z+stepZ))) / -stepX,
                //                               Private::affectRGB(voxels(VecN<3,int>(x+stepX,y+2*stepY,z+stepZ)),voxels(VecN<3,int>(x+stepX,y+stepY,z+stepZ))) / -stepY,
                //                               Private::affectRGB(voxels(VecN<3,int>(x+stepX,y+stepY,z+2*stepZ)),voxels(VecN<3,int>(x+stepX,y+stepY,z+stepZ))) / -stepZ
                //                              },
                //                              {x,y+stepY,z+stepZ,
                //                               Private::affectRGB(voxels(VecN<3,int>(x+stepX,y+stepY,z+stepZ)),voxels(VecN<3,int>(x,y+stepY,z+stepZ))) / -stepX,
                //                               Private::affectRGB(voxels(VecN<3,int>(x,y+2*stepY,z+stepZ)),    voxels(VecN<3,int>(x,y+stepY,z+stepZ))) / -stepY,
                //                               Private::affectRGB(voxels(VecN<3,int>(x,y+stepY,z+2*stepZ)),    voxels(VecN<3,int>(x,y+stepY,z+stepZ))) / -stepZ
                //                              }
                //                          },{
                //                              voxels(VecN<3,int>(x,y,z)).lumi(),
                //                              voxels(VecN<3,int>(x+stepX,y,z)).lumi(),
                //                              voxels(VecN<3,int>(x+stepX,y,z+stepZ)).lumi(),
                //                              voxels(VecN<3,int>(x,y,z+stepZ)).lumi(),
                //                              voxels(VecN<3,int>(x,y+stepY,z)).lumi(),
                //                              voxels(VecN<3,int>(x+stepX,y+stepY,z)).lumi(),
                //                              voxels(VecN<3,int>(x+stepX,y+stepY,z+stepZ)).lumi(),
                //                              voxels(VecN<3,int>(x,y+stepY,z+stepZ)).lumi()
                //                          }};
                processCube(c, vertexList,cc);

            }
        }
    }

    return vertexList;
}
std::vector<Visualization::vertex > Visualization::runMarchingCubes2(const MatN<3,F64 > &voxel,F32 isosurface) {

    MatN<3,F32 > voxels(voxel.getDomain()+4);
    voxels=-1;
    MatN<3,F32 >::IteratorEDomain it (voxel.getIteratorEDomain());


    while(it.next())
    {
        VecN<3,int> x  =it.x();
        x=x+2;
        voxels(x)=voxel(it.x());
    }

    std::vector<vertex > vertexList;
    int sizeX =  voxels.getDomain()(0);
    int sizeY=voxels.getDomain()(1);
    int sizeZ=voxels.getDomain()(2);
    int stepX=1;
    int stepY=1;
    int stepZ=1;


    // Run the processCube function on every cube in the grid
    for(pop::F32 x = stepX; x < sizeX-2*stepX; x += stepX) {
        for(pop::F32 y = stepY; y < sizeY-2*stepY; y += stepY) {
            for(pop::F32 z = stepZ; z < sizeZ-2*stepZ; z += stepZ) {
                cubeF c = {{
                               {x,y,z,
                                (F32)(voxels(VecN<3,int>(x+stepX,y,z))-voxels(VecN<3,int>(x-stepX,y,z))) / -stepX,
                                (F32)(voxels(VecN<3,int>(x,y+stepY,z))-voxels(VecN<3,int>(x,y-stepY,z))) / -stepY,
                                (F32)(voxels(VecN<3,int>(x,y,z+stepZ))-voxels(VecN<3,int>(x,y,z-stepZ))) / -stepZ
                               },
                               {x+stepX,y,z,
                                (F32)(voxels(VecN<3,int>(x+2*stepX,y,z))-    voxels(VecN<3,int>(x,y,z))) / -stepX,
                                (F32)(voxels(VecN<3,int>(x+stepX,y+stepY,z))-voxels(VecN<3,int>(x+stepX,y-stepY,z))) / -stepY,
                                (F32)(voxels(VecN<3,int>(x+stepX,y,z+stepZ))-voxels(VecN<3,int>(x+stepX,y,z-stepZ))) / -stepZ
                               },
                               {x+stepX,y,z+stepZ,
                                (F32)(voxels(VecN<3,int>(x+2*stepX,y,z+stepZ))-    voxels(VecN<3,int>(x,y,z+stepZ))) / -stepX,
                                (F32)(voxels(VecN<3,int>(x+stepX,y+stepY,z+stepZ))-voxels(VecN<3,int>(x+stepX,y-stepY,z+stepZ))) / -stepY,
                                (F32)(voxels(VecN<3,int>(x+stepX,y,z+2*stepZ))-    voxels(VecN<3,int>(x+stepX,y,z))) / -stepZ
                               },
                               {x,y,z+stepZ,
                                (F32)(voxels(VecN<3,int>(x+stepX,y,z+stepZ))-voxels(VecN<3,int>(x-stepX,y,z+stepZ))) / -stepX,
                                (F32)(voxels(VecN<3,int>(x,y+stepY,z+stepZ))-voxels(VecN<3,int>(x,y-stepY,z+stepZ))) / -stepY,
                                (F32)(voxels(VecN<3,int>(x,y,z+2*stepZ))-   voxels(VecN<3,int>(x,y,z))) / -stepZ
                               },
                               {x,y+stepY,z,
                                (F32)(voxels(VecN<3,int>(x+stepX,y+stepY,z))-voxels(VecN<3,int>(x-stepX,y+stepY,z))) / -stepX,
                                (F32)(voxels(VecN<3,int>(x,y+2*stepY,z))-voxels(VecN<3,int>(x,y,z))) / -stepY,
                                (F32)(voxels(VecN<3,int>(x,y+stepY,z+stepZ))-voxels(VecN<3,int>(x,y+stepY,z-stepZ))) / -stepZ
                               },
                               {x+stepX,y+stepY,z,
                                (F32)(voxels(VecN<3,int>(x+2*stepX,y+stepY,z))-voxels(VecN<3,int>(x,y+stepY,z))) / -stepX,
                                (F32)(voxels(VecN<3,int>(x+stepX,y+2*stepY,z))-voxels(VecN<3,int>(x+stepX,y,z))) / -stepY,
                                (F32)(voxels(VecN<3,int>(x+stepX,y+stepY,z+stepZ))-voxels(VecN<3,int>(x+stepX,y+stepY,z-stepZ))) / -stepZ
                               },
                               {x+stepX,y+stepY,z+stepZ,
                                (F32)(voxels(VecN<3,int>(x+2*stepX,y+stepY,z+stepZ))-voxels(VecN<3,int>(x,y+stepY,z+stepZ))) / -stepX,
                                (F32)(voxels(VecN<3,int>(x+stepX,y+2*stepY,z+stepZ))-voxels(VecN<3,int>(x+stepX,y,z+stepZ))) / -stepY,
                                (F32)(voxels(VecN<3,int>(x+stepX,y+stepY,z+2*stepZ))-voxels(VecN<3,int>(x+stepX,y+stepY,z))) / -stepZ
                               },
                               {x,y+stepY,z+stepZ,
                                (F32)(voxels(VecN<3,int>(x+stepX,y+stepY,z+stepZ))-voxels(VecN<3,int>(x-stepX,y+stepY,z+stepZ))) / -stepX,
                                (F32)(voxels(VecN<3,int>(x,y+2*stepY,z+stepZ))-voxels(VecN<3,int>(x,y,z+stepZ))) / -stepY,
                                (F32)(voxels(VecN<3,int>(x,y+stepY,z+2*stepZ))-voxels(VecN<3,int>(x,y+stepY,z))) / -stepZ
                               }
                           },{
                               static_cast<pop::F32>(voxels(VecN<3,int>(x,y,z))),
                               static_cast<pop::F32>(voxels(VecN<3,int>(x+stepX,y,z))),
                               static_cast<pop::F32>(voxels(VecN<3,int>(x+stepX,y,z+stepZ))),
                               static_cast<pop::F32>(voxels(VecN<3,int>(x,y,z+stepZ))),
                               static_cast<pop::F32>(voxels(VecN<3,int>(x,y+stepY,z))),
                               static_cast<pop::F32>(voxels(VecN<3,int>(x+stepX,y+stepY,z))),
                               static_cast<pop::F32>(voxels(VecN<3,int>(x+stepX,y+stepY,z+stepZ))),
                               static_cast<pop::F32>(voxels(VecN<3,int>(x,y+stepY,z+stepZ)))
                           }};
                processCube(c, vertexList,isosurface,true);

            }
        }
    }

    return vertexList;
}

void Visualization::axis(Scene3d &scene, double length, double width, double trans_minus){

    trans_minus = -trans_minus;
    FigureLine * line = new FigureLine;
    line->width =  width;
    line->x1=Vec3F64(trans_minus,trans_minus,trans_minus);
    line->x2=Vec3F64(length,trans_minus,trans_minus);
    line->setRGB(RGBUI8(255,0,0));
    scene._v_figure.push_back(line);
    FigureCone * cone = new FigureCone;
    cone->setRGB(RGBUI8(255,0,0));
    cone->x = Vec3F64(length,trans_minus,trans_minus);
    cone->dir = Vec3F64(1,0,0);
    cone->h = length/4;
    cone->r = width;
    scene._v_figure.push_back(cone);

    line = new FigureLine;
    line->width =  width;
    line->x1=Vec3F64(trans_minus,trans_minus,trans_minus);
    line->x2=Vec3F64(trans_minus,length,trans_minus);
    line->setRGB(RGBUI8(0,255,0));
    scene._v_figure.push_back(line);
    cone = new FigureCone;
    cone->setRGB(RGBUI8(0,255,0));
    cone->x = Vec3F64(trans_minus,length,trans_minus);
    cone->dir = Vec3F64(0,1,0);
    cone->h = length/4;
    cone->r = width;
    scene._v_figure.push_back(cone);

    line = new FigureLine;
    line->width =  width;
    line->x1=Vec3F64(trans_minus,trans_minus,trans_minus);
    line->x2=Vec3F64(trans_minus,trans_minus,length);
    line->setRGB(RGBUI8(0,0,255));
    scene._v_figure.push_back(line);
    cone = new FigureCone;
    cone->setRGB(RGBUI8(0,0,255));
    cone->x = Vec3F64(trans_minus,trans_minus,length);
    cone->dir = Vec3F64(0,0,1);
    cone->h = length/4;
    cone->r = width;
    scene._v_figure.push_back(cone);

}

}
