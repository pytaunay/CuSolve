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
#include <equation_system/clientp.hpp>

namespace System {
using namespace std;
using namespace BOOST_SPIRIT_CLASSIC_NS;
namespace clientp
{

namespace {
	void sad_little_message(int line_no, string input){
		cerr << "Input line : " << line_no << endl; 
		cerr << "Received : " << input << endl; 
		cerr << "Please express equations using arithmetic combinations +/- of one or more terms, " << endl;
		cerr << "terms consisting of products expressed with *, powers with ^ and powers/indices" << endl;
		cerr << "expressed within parenthesis for constants k and variables y (1-based index) eg.," << endl;
		cerr << " " << endl;
		cerr << "1.0*k(717)*y(516)-1.0*k(416)*y(1)*y(392)-1.0*k(718)*y(1)^(1/2)*y(517)" << endl;
		cerr << " " << endl;
		throw std::invalid_argument( "Received bad equation format on stdin" );
	}
}


template <typename T>
void update(std::map<T,T> &x, const T& key, const T& val){

	x[key]+=val;
}


namespace qi = boost::spirit::qi;
namespace ascii = boost::spirit::ascii;
namespace phoenix = boost::phoenix;

using qi::float_;
using qi::double_;
using qi::phrase_parse;
using qi::_1;
using ascii::space;
using phoenix::push_back;


template <typename Iterator>
	bool parse_csv(Iterator first, Iterator last, std::vector<float>& k)
	{
		bool r = phrase_parse(first, last,

				//  grammar for csv files
				(
				 //*(float_ >> ',' | float_ [push_back(phoenix::ref(k),_1)])
				 *(float_ [push_back(phoenix::ref(k),_1)])
				)
				,
				space);

		return r;

	}


template <typename Iterator>
	bool parse_k_vals(Iterator first, Iterator last, std::vector<float>& w)
	{			
		bool r = phrase_parse(first, last,

				//  grammar for k constants
				(
				 ("k(" >> float_[push_back(phoenix::ref(w), _1)] >> ')')
				)
				,
				//skip
				space | '*' | float_ >> '*' | '-'>>float_ >> '*'| '+' >> float_ >> '*'| "y(" >> float_ >> ')');

		return r;
	}

template <typename Iterator>
	bool parse_constants(Iterator first, Iterator last, std::vector<float>& v)
	{
		bool r = phrase_parse(first, last,

				//  grammar for numerical constants & sign
				(
				 (float_[push_back(phoenix::ref(v), _1)] >> '*')
				)
				,
				//skip
				space);
		return r;
	}
template <typename Iterator>
	bool parse_y_vals(Iterator first, Iterator last, std::map<float,float>& x)
	{
		float base =1.0;
		float exp = 0.0;
		bool r = phrase_parse(first, last,

				//  grammar for y variables
				//  this looks like hell
				(
				 *("y(" >> float_[phoenix::ref(base)=_1] >> ")^(" >> float_[phoenix::ref(exp)=_1] >> '/' >> float_[phoenix::ref(exp)/=_1]\
					 [boost::bind(&update<float>,boost::ref(x),boost::ref(base),boost::ref(exp))] >> ')' |
					 "y(" >> float_[phoenix::ref(base)=_1, boost::bind(&update<float>,boost::ref(x),boost::ref(base),1.0f)] >> ')')
				)
				,
				//skip
				space | '*' | float_ >> '*' | '-'>>float_ >> '*'| '+' >> float_ >> '*'| "k(" >> float_ >> ')');


		return r;
	}

template <typename Iterator>
	bool parse_csv(Iterator first, Iterator last, std::vector<double>& k)
	{
		bool r = phrase_parse(first, last,

				//  grammar for csv files
				(
				 //*(double_ >> ',' | double_ [push_back(phoenix::ref(k),_1)])
				 *(double_ [push_back(phoenix::ref(k),_1)])
				)
				,
				space);

		return r;

	}


template <typename Iterator>
	bool parse_k_vals(Iterator first, Iterator last, std::vector<double>& w)
	{			
		bool r = phrase_parse(first, last,

				//  grammar for k constants
				(
				 ("k(" >> double_[push_back(phoenix::ref(w), _1)] >> ')')
				)
				,
				//skip
				space | '*' | double_ >> '*' | '-'>>double_ >> '*'| '+' >> double_ >> '*'| "y(" >> double_ >> ')');

		return r;
	}

template <typename Iterator>
	bool parse_constants(Iterator first, Iterator last, std::vector<double>& v)
	{
		bool r = phrase_parse(first, last,

				//  grammar for numerical constants & sign
				(
				 (double_[push_back(phoenix::ref(v), _1)] >> '*')
				)
				,
				//skip
				space);
		return r;
	}
template <typename Iterator>
	bool parse_y_vals(Iterator first, Iterator last, std::map<double,double>& x)
	{
		double base =1.0;
		double exp = 0.0;
		bool r = phrase_parse(first, last,

				//  grammar for y variables
				//  this looks like hell
				(
				 *("y(" >> double_[phoenix::ref(base)=_1] >> ")^(" >> double_[phoenix::ref(exp)=_1] >> '/' >> double_[phoenix::ref(exp)/=_1]\
					 [boost::bind(&update<double>,boost::ref(x),boost::ref(base),boost::ref(exp))] >> ')' |
					 "y(" >> double_[phoenix::ref(base)=_1, boost::bind(&update<double>,boost::ref(x),boost::ref(base),1.0f)] >> ')')
				)
				,
				//skip
				space | '*' | double_ >> '*' | '-'>>double_ >> '*'| '+' >> double_ >> '*'| "k(" >> double_ >> ')');


		return r;
	}
}
}
