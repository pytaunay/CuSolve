/**
 * @file evalnode.h
 * @author Pierre-Yves Taunay (py.taunay@psu.edu)
 * @date November, 2013
 * @brief Simple struct for a node in an equation tree
 *
 */

#pragma once

namespace System {
	template<typename T>
	struct EvalNode{

		T constant;
		int kIdx;
		int yIdx1;
		T yExp1;
		int yIdx2;
		T yExp2;

	};
}

