#pragma once

#include <cmath>
#include <iostream>
#include <cstdio>

#include <stdint.h>

#include <cstdlib>
#include <cstring>


namespace utils {
	namespace {
		__host__ void dbl2bin(double &d) {
			uint64_t u;
			memcpy(&u,&d,sizeof(d));

			for ( int i = 63; i >= 0; i-- )
			{
				printf( "%d", (u >> i ) & 1 );
				if(i == 63) {
					printf(" ");
				} else if (i==52){	
					printf(" ");
				}	
			}
		}	
	}
}
