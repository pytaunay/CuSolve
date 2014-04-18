#pragma once

#include <boost/config/warning_disable.hpp>
#include <boost/bind.hpp>
#include <boost/spirit/include/qi.hpp>
#include <boost/spirit/include/qi_real.hpp>
#include <boost/spirit/include/phoenix_core.hpp>
#include <boost/spirit/include/phoenix_operator.hpp>
#include <boost/spirit/include/phoenix_stl.hpp>
#include <boost/spirit/include/classic_core.hpp>
#include <boost/spirit/include/classic_increment_actor.hpp>
#include <string>
#include <map>
#include <iostream>
#include <fstream>
#include <stdexcept>


namespace System {
using namespace std;
using namespace BOOST_SPIRIT_CLASSIC_NS;
	namespace clientp
	{


		template <typename T>
		void update(std::map<T,T> &x, const T& key, const T& val);

		template <typename Iterator>
			bool parse_csv(Iterator first, Iterator last, std::vector<float>& k);
		
		template <typename Iterator>
			bool parse_k_vals(Iterator first, Iterator last, std::vector<float>& w);

		template <typename Iterator>
			bool parse_constants(Iterator first, Iterator last, std::vector<float>& v);

		template <typename Iterator>
			bool parse_y_vals(Iterator first, Iterator last, std::map<float,float>& x);
			
		template <typename Iterator>
			bool parse_csv(Iterator first, Iterator last, std::vector<double>& k);

		template <typename Iterator>
			bool parse_k_vals(Iterator first, Iterator last, std::vector<double>& w);

		template <typename Iterator>
			bool parse_constants(Iterator first, Iterator last, std::vector<double>& v);
			
		template <typename Iterator>
			bool parse_y_vals(Iterator first, Iterator last, std::map<double,double>& x);
	}
}

#include <equation_system/detail/clientp.inl>
