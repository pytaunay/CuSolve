/**
 * @file odesolver.h
 * @author Pierre-Yves Taunay (py.taunay@psu.edu)
 * @date March, 2014
 * @brief Abstract representation of common classes of ODE solvers
 *
 */

#pragma once

namespace NumericalSolver {
	
	/*! \brief Empty parent class for ODEs
	 *
	 * This is an empty parent class from which all ODE solvers will be derived.
	 * Allows for modularity: each ODE solver can derive from that class.
	 */
	class ODESolver {};

	/*! \brief Empty parent class for implicit methods
	 *
	 */
	class ImplicitODESolver : public ODESolver {};

	/*! \brief Empty parent class for explicit methods
	 *
	 */
	class ExplicitODESolver : public ODESolver {};
	
	/*! \brief Empty parent class for linear multistep methods
	 *
	 */
	class LMMODESolver : public ODESolver {};

	/*! \brief Empty parent class for Runge Kutta methods
	 *
	 */
	class RKODESolver : public ODESolver {};

}

