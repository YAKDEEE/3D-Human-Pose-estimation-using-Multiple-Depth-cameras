#pragma once

//
// DTW.hpp
//
// Copyright (c) 2019 Charles Jekel
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//


#include <cstdlib>
#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include<numeric>
const std::string DTW_VERSION = "0.0.1";
double min_a = 10e-6;
double pi = 3.141592;
//KINECT->FBX (joint 25)
int i_KIN2FBX[28] = { 0 ,6, 9 , 12, 13, 16, 18 ,20 ,22 ,999 ,
999 , 14 , 17 ,19 ,21 , 23 , 999 , 999 ,1 ,4 ,7 ,10,2 ,5 ,8 ,11 ,24 , 15 };

using namespace std;
namespace DTW
{
    /**
     * Compute the p_norm between two 1D c++ vectors.
     *
     * The p_norm is sometimes referred to as the Minkowski norm. Common
     * p_norms include p=2.0 for the euclidean norm, or p=1.0 for the
     * manhattan distance. See also
     * https://en.wikipedia.org/wiki/Norm_(mathematics)#p-norm
     *
     * @a 1D vector of m size, where m is the number of dimensions.
     * @b 1D vector of m size (must be the same size as b).
     * @p value of norm to use.
     */
    double p_norm(vector<vector<double>> a, vector<vector<double>> b, double p) {
        double d = 0;
        double high_weight = 1;
        double middle_weight = 0.9;
        double low_weight = 0.8;
        double super_low_weight = 1;
        //28-11 개
        for (int joint_index = 0; joint_index <= 27; joint_index++) {
            if (joint_index == 9 || joint_index == 16 || joint_index == 17 || joint_index == 10 || joint_index == 0) { continue; } //not using
            if (joint_index == 26 || joint_index == 27) { continue; }

            int joint_weight = 1.0;
            if (joint_index == 7 || joint_index == 14 || joint_index == 24 || joint_index == 20 || joint_index == 6 || joint_index == 13 || joint_index == 23 || joint_index == 19) { joint_weight = high_weight; }
            else if (joint_index == 12 || joint_index == 11 || joint_index == 4 || joint_index == 5) { joint_weight = high_weight; }
            else if (joint_index == 3 || joint_index == 2 || joint_index == 1 || joint_index == 18 || joint_index == 22) { joint_weight = high_weight; }
            else if (joint_index == 8 || joint_index == 15 || joint_index == 21 || joint_index == 25) { joint_weight = high_weight; }

            d += pow(joint_weight * abs(a[joint_index][0] - b[joint_index][0]), p);
            d += pow(joint_weight * abs(a[joint_index][1] - b[joint_index][1]), p);
            d += pow(joint_weight * abs(a[joint_index][2] - b[joint_index][2]), p);

        }
        return pow(d, 1.0 / p);
    };

    double p_norm(std::vector<double> a, std::vector<double> b, double p) { //github원본
        double d = 0;
        for (int i = 0; i < a.size(); i++) {
            d += std::pow(std::abs(a[i] - b[i]), p);
        }
        return std::pow(d, 1.0 / p);
    };
    double inner_prod(vector<vector<double>> a, vector<vector<double>> b) {
        double d = 0;
        int joint_index = 0;


        for (int joint_index = 0; joint_index <= 27; joint_index++) {
            double sum = 0;
            if (joint_index == 9 || joint_index == 16 || joint_index == 17 || joint_index == 10 || joint_index == 0) { continue; } //not using
            if (joint_index == 5 || joint_index == 12) { continue; } //not using 쇄골각도
            if (joint_index == 26 || joint_index == 27) { continue; }

            double a_div = sqrt(pow(a[joint_index][0], 2) + pow(a[joint_index][1], 2) + pow(a[joint_index][2], 2));
            double b_div = sqrt(pow(b[joint_index][0], 2) + pow(b[joint_index][1], 2) + pow(b[joint_index][2], 2));
            sum += a[joint_index][0] * b[joint_index][0];
            sum += a[joint_index][1] * b[joint_index][1];
            sum += a[joint_index][2] * b[joint_index][2];
            if (a_div < min_a || b_div < min_a) {
                sum = 0;
            }
            else {
                sum = abs(sum / a_div / b_div);
            }
            d += 1-sum;
        }
        return  d;
    };

    double several_inner_prod(vector<vector<double>> a, vector<vector<double>> b, int joint_index) {
            double d = 0;

            double sum = 0;
            double a_div = sqrt(pow(a[joint_index][0], 2) + pow(a[joint_index][1], 2) + pow(a[joint_index][2], 2));
            double b_div = sqrt(pow(b[joint_index][0], 2) + pow(b[joint_index][1], 2) + pow(b[joint_index][2], 2));

            sum += a[joint_index][0] * b[joint_index][0];
            sum += a[joint_index][1] * b[joint_index][1];
            sum += a[joint_index][2] * b[joint_index][2];
            if (a_div < min_a || b_div < min_a) {
                sum = 0;
            }
            else {
                sum = abs(sum / a_div / b_div);
            }
            d += 1-sum;
        
        return  d;
    };
    
    /**
     * Compute the DTW distance between two 2D c++ vectors.
     *
     * The c++ vectors can have different number of data points, but must
     * have the same number of dimensions. This will raise
     * std::invalid_argument if the dimmensions of a and b are different.
     * Here the vectors should be formatted as
     * [number_of_data_points][number_of_dimensions]. The DTW distance can
     * be computed for any p_norm. See the wiki for more DTW info.
     * https://en.wikipedia.org/wiki/Dynamic_time_warping
     *
     * @a 2D vector of [number_of_data_points][number_of_dimensions].
     * @b 2D vector of [number_of_data_points][number_of_dimensions].
     * @p value of p_norm to use.
     */
    double dtw_distance_only(vector<vector<vector<double>>> a,
        vector<vector<vector<double>>> b, vector<vector<double>> &ans,
        double p)
    {
        int n = a.size();
        int o = b.size();
        int a_m = a[0].size();
        int b_m = b[0].size();
        if (a_m != b_m)
        {
            throw std::invalid_argument("a and b must have the same number of dimensions!");
        }
        vector<vector<double>> d(n, vector<double>(o, 0.0));
        d[0][0] = p_norm(a[0], b[0], p);
        for (int i = 1; i < n; i++)
        {
            d[i][0] = d[i - 1][0] + p_norm(a[i], b[0], p);
        }
        for (int i = 1; i < o; i++)
        {
            d[0][i] = d[0][i - 1] + p_norm(a[0], b[i], p);
        }
        for (int i = 1; i < n; i++)
        {
            for (int j = 1; j < o; j++) {
                d[i][j] = p_norm(a[i], b[j], p) + std::fmin(std::fmin(d[i - 1][j], d[i][j - 1]), d[i - 1][j - 1]);
            }
        }
        ans = d;
        return d[n - 1][o - 1];
    };

    double dtw_distance_only2(vector<vector<vector<double>>> a,
        vector<vector<vector<double>>> b, vector<vector<double>>& ans
    )
    {
        
        int n = a.size();
        int o = b.size();
        int a_m = a[0].size();
        int b_m = b[0].size();
        if (a_m != b_m)
        {
            throw std::invalid_argument("a and b must have the same number of dimensions!");
        }
        std::vector<std::vector<double> > d(n, std::vector<double>(o, 0));  

        d[0][0] = inner_prod(a[0], b[0]);
        for (int i = 1; i < n; i++)
        {
            d[i][0] = d[i - 1][0] + inner_prod(a[i], b[0]);
        }
        for (int i = 1; i < o; i++)
        {
            d[0][i] = d[0][i - 1] + inner_prod(a[0], b[i]);


            
        }
        for (int i = 1; i < n; i++)
        {
            for (int j = 1; j < o; j++) {
                d[i][j] = inner_prod(a[i], b[j]) + std::fmin(std::fmin(d[i - 1][j], d[i][j - 1]), d[i - 1][j - 1]);
            }
        }
        ans = d;
        return d[n - 1][o - 1];
    };

	double dtw_distance_only2_several_joints(vector<vector<vector<double>>> a,
		vector<vector<vector<double>>> b, int index
	)
	{
		int n = a.size();
		int o = b.size();
		int a_m = a[0].size();
		int b_m = b[0].size();
        
		if (a_m != b_m)
		{
			throw std::invalid_argument("a and b must have the same number of dimensions!");
		}
		std::vector<std::vector<double> > d(n, std::vector<double>(o, 0));

		d[0][0] = several_inner_prod(a[0], b[0],index);
		for (int i = 1; i < n; i++)
		{
			d[i][0] = d[i - 1][0] + several_inner_prod(a[i], b[0],index);
		}
		for (int i = 1; i < o; i++)
		{
			d[0][i] = d[0][i - 1] + several_inner_prod(a[0], b[i],index);



		}
		for (int i = 1; i < n; i++)
		{
			for (int j = 1; j < o; j++) {
				d[i][j] = several_inner_prod(a[i], b[j],index) + std::fmin(std::fmin(d[i - 1][j], d[i][j - 1]), d[i - 1][j - 1]);
			}
		}
        
		return d[n - 1][o - 1];
	};
    //2023-11-16 dtw path
    std::vector<std::vector<int> > dtw_path(vector<vector<double>>dtw_vector, vector<int>& aligned_path) {
        int i = dtw_vector.size()-1 ;
		int j = dtw_vector[0].size()-1 ;
        vector<std::vector<int> > my_path = { {i, j} };
        aligned_path[j] = i;
        while (i > 0 || j > 0) {

            if (i == 0)
            {
                j -= 1;
            }
            else if (j == 0)
            {
                i -= 1;
            }
            else
            {
                double temp_step = std::fmin(std::fmin(dtw_vector[i - 1][j], dtw_vector[i][j - 1]),
                    dtw_vector[i - 1][j - 1]);
                if (temp_step == dtw_vector[i - 1][j])
                {
                    i -= 1;
                }
                else if (temp_step == dtw_vector[i][j - 1])
                {
                    j -= 1;
                }
                else
                {
                    i -= 1;
                    j -= 1;
                }
				//aligned_path[j] = i;
            }
			aligned_path[j] = i;
            my_path.push_back({ i, j });
        }
        std::reverse(my_path.begin(), my_path.end());
        
        return my_path;
    };
    //
        /**
         * Assembles a 2D c++ DTW distance vector.
         *
         * The DTW distance vector represents the matrix of DTW distances for
         * all possible alignments. The c++ vectors must have the same 2D size.
         * d.size() == c.size() == number of a data points, where d[0].size ==
         * c[0].size() == number of b data points.
         *
         * @d 2D DTW distance vector of [a data points][b data points].
         * @c 2D pairwise distance vector between every a and b data point.
         */
        vector<vector<double>> dtw_vector_assemble(vector<vector<double>> d, vector<vector<double>> c) { 
            int n = d.size();
            int o = d[0].size();
            for (int i = 1; i < n; i++)
            {
                d[i][0] = d[i - 1][0] + c[i][0];
            }
            for (int i = 1; i < o; i++)
            {
                d[0][i] = d[0][i - 1] + c[0][i];
            }
            for (int i = 1; i < n; i++)
            {
                for (int j = 1; j < o; j++) {
                    d[i][j] = c[i][j] + std::fmin(std::fmin(d[i - 1][j], d[i][j - 1]), d[i - 1][j - 1]);
                }
            }
            return d;

        };


        class DTW {

        public:
            std::vector<std::vector<double> > a_vector, b_vector;
            int a_data, n_dim, b_data;
            double p, distance;
            std::vector<std::vector<double> > dtw_vector, pairwise_vector;

            /**
              * Class for Dynamic Time Warping distance between two 2D c++ vectors.
              *
              * The c++ vectors can have different number of data points, but must
              * have the same number of dimensions. This will raise
              * std::invalid_argument if the dimmensions of a and b are different.
              * Here the vectors should be formatted as
              * [number_of_data_points][number_of_dimensions]. The DTW distance can
              * be computed for any p_norm. See the wiki for more DTW info.
              * https://en.wikipedia.org/wiki/Dynamic_time_warping
              *
              * @a 2D vector of [number_of_data_points][number_of_dimensions].
              * @b 2D vector of [number_of_data_points][number_of_dimensions].
              * @p value of p_norm to use.
              *
              * This class stores the following:
              *
              * @DTW.distance Computed DTW distance.
              * @DTW.pairwise_vector P_norm distance between each a and b data point.
              * @DTW.dtw_vector DTW distance matrix.
              *
              * The class has the following methods:
              *
              * @DTW.path() Returns a 2D vector of the alignment path between a and b.
              */
            DTW(std::vector<std::vector<double> > a, std::vector<std::vector<double> > b, double p) :
                a_vector(a), b_vector(b), p(p) {
                a_data = a.size();
                b_data = b.size();
                int a_m = a_vector[0].size();
                int b_m = b_vector[0].size();

                if (a_m != b_m)
                {
                    throw std::invalid_argument("a and b must have the same number of dimensions!");
                }
                else
                {
                    n_dim = a_m;
                }

                std::vector<std::vector<double> > c(a_data, std::vector<double>(b_data, 0.0));

                //for (int i = 0; i < a_data; i++) {
                //    for (int j = 0; j < b_data; j++) {
                //        c[i][j] = p_norm(a_vector[i], b_vector[j], p);
                //    }
                //}
                pairwise_vector = c;
                std::vector<std::vector<double> > d(a_data, std::vector<double>(b_data, 0.0));
                d[0][0] = pairwise_vector[0][0];
                dtw_vector = dtw_vector_assemble(d, pairwise_vector);
                distance = dtw_vector[a_data - 1][b_data - 1];
            };

            /**
             * Returns a 2D vector of the alignment path between a and b.
             *
             * The DTW path is a 2D integer vector, where [path_length][i] represents
             * the i'th data point on curve a, and [path_length][j] represents the j'th
             * data point on curve b. The path_length will depend upon the optimal DTW
             * alignment.
             */
            std::vector<std::vector<int> > path() {
                int i = a_data - 1;
                int j = b_data - 1;
                std::vector<std::vector<int> > my_path = { {i, j} };
                while (i > 0 || j > 0) {
                    if (i == 0)
                    {
                        j -= 1;
                    }
                    else if (j == 0)
                    {
                        i -= 1;
                    }
                    else
                    {
                        double temp_step = std::fmin(std::fmin(dtw_vector[i - 1][j], dtw_vector[i][j - 1]),
                            dtw_vector[i - 1][j - 1]);
                        if (temp_step == dtw_vector[i - 1][j])
                        {
                            i -= 1;
                        }
                        else if (temp_step == dtw_vector[i][j - 1])
                        {
                            j -= 1;
                        }
                        else
                        {
                            i -= 1;
                            j -= 1;
                        }
                    }
                    my_path.push_back({ i, j });
                }
                std::reverse(my_path.begin(), my_path.end());
                return my_path;
            }
        };

    }
#pragma once
