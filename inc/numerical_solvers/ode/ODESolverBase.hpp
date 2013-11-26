#ifndef _ODESOLVERBASE_HPP_
#define _ODESOLVERBASE_HPP_

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

#endif

