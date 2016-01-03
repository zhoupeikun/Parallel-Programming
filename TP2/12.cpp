#include <stdio.h>
#include <stdlib.h>

#include <x86intrin.h>
#include <iostream>

#include <algorithm>
#include <functional>
#include <time.h>
#include <random>
#include <vector>
#include <numeric>
#include <chrono>
#include <omp.h>

// Defini la taille de l'image pour benchmandelbrot
#define SIZE 200

// Redefinition of a 128 bits simd vector
typedef __m128 vfloat32;
typedef __m128i vuint32;

// Macro pour faire les décalages
#define vect_left1(v0,v1) _mm_shuffle_ps(_mm_shuffle_ps(v0,v1,_MM_SHUFFLE(0,0,3,3)),v1,_MM_SHUFFLE(2,1,3,1))

#define vect_right1(v0,v1) _mm_shuffle_ps(v0,_mm_shuffle_ps(v0,v1,_MM_SHUFFLE(0,0,3,3)),_MM_SHUFFLE(3,1,2,1))

// function pour avoir le nombre de cycles
uint64_t rdtsc(){
	return __rdtsc();
}

// -------------------------------------------------------------------
void vectoradd_simd( vfloat32 *vX1, vfloat32 *vX2, vfloat32 *vY , size_t size)
// -------------------------------------------------------------------
{
	vfloat32 x1, x2, y;

	for(size_t i=0; i< size; ++i) {

		x1 = _mm_load_ps((float*) &vX1[i]);
		x2 = _mm_load_ps((float*) &vX2[i]);

		y = _mm_add_ps(x1, x2);

		_mm_store_ps((float*) &vY[i], y);
	}
}

// ---------------------------------------------------------
vfloat32 vectordot_simd(vfloat32 *vX1, vfloat32 *vX2, size_t size)
// ---------------------------------------------------------
{
	vfloat32 x1, x2, p, s;
	s = _mm_setzero_ps();

	for(size_t i=0; i<size; ++i){

		x1 = _mm_load_ps((float*) &vX1[i]);
		x2 = _mm_load_ps((float*) &vX2[i]);
		p = _mm_mul_ps(x1,x2);
		s = _mm_add_ps(s,p);

	}
	x1 = _mm_shuffle_ps(s,s, _MM_SHUFFLE(1,0,3,2));
	x1 = _mm_add_ps(x1,s);
	s = _mm_shuffle_ps(x1,x1, _MM_SHUFFLE(0,1,2,3));
	s = _mm_add_ps(s,x1);

  return s; // attention il faut retourner un registre SIMD et non un scalaire
}

// ----------------------------------------------------
void vectoravg3_simd(vfloat32 *vX, vfloat32 *vY , size_t size)
// ----------------------------------------------------
{
	vfloat32 x1,x2,x3,divide;
	divide = _mm_set1_ps(1.0/3.0);

	for(size_t i=1 ; i<size-1 ; ++i){

		x1 = _mm_load_ps((float*) &vX[i-1]);
		x2 = _mm_load_ps((float*) &vX[i]);
		x3 = _mm_load_ps((float*) &vX[i+1]);

		x1 = vect_left1(x1,x2);
		x3 = vect_right1(x2,x3);

		vY[i] = _mm_mul_ps(divide, _mm_add_ps(x3, _mm_add_ps(x2,x1)));
	}
}

// --------------------------------------------------
size_t mandelbrot_scalar(float a, float b, size_t max_iter)
// --------------------------------------------------
{
	
        float x=0;
        float y=0;
        float z;
        float tmp;
        size_t num_iter=0;
        while(num_iter<max_iter){
		tmp=x;
		x=(x*x)-(y*y)+a;
		y=2*y*tmp+b;
		z=sqrt((x*x)+(y*y));
		if(z>=2) {break;}
		num_iter++;
}
	return num_iter;
}


// --------------------------------------------------------------
vuint32 mandelbrot_simd(vfloat32 a, vfloat32 b, size_t max_iter)
// --------------------------------------------------------------
{
	vuint32 num_iter = _mm_set1_epi32(0);
	vfloat32 zero=_mm_set1_ps(0);
    vfloat32 one=_mm_set1_ps(1);
	vfloat32 two=_mm_set1_ps(2);
	vfloat32 x=_mm_setzero_ps();
	vfloat32 y=_mm_setzero_ps();
	vfloat32 tmp;
	vfloat32 z;
	for(int i=0;i<max_iter;i++){	
	tmp=x;
	x=_mm_add_ps(a,_mm_sub_ps(_mm_mul_ps(x,x),_mm_mul_ps(y,y)));
	y=_mm_add_ps(b,_mm_mul_ps(_mm_mul_ps(y,tmp),two));	
	z=_mm_sqrt_ps(_mm_add_ps(_mm_mul_ps(x,x),_mm_mul_ps(y,y)));
	num_iter=_mm_add_epi32(num_iter,_mm_cvtps_epi32(_mm_and_ps(_mm_cmplt_ps(z,two),one)));
    }
	return num_iter;
}

// --------------------------------------------------------------------------------------------------------
void calc_mandelbrot_scalar(std::vector<size_t> & M, size_t h, size_t w, float a0, float a1, float b0, float b1, size_t max_iter)
// --------------------------------------------------------------------------------------------------------
{
    // intervale de valeurs: [a0:a1]x[b0:b1]

    // la seule chose a modifier dans cette fonction est la ligne de pragma OpenMP	
    
	size_t i, j;

	float da, db;
	float a, b;
	size_t iter;

	da = (a1-a0)/w;
	db = (b1-b0)/h;
    //#pragma omp parallel for schedule(dynamic, 1) 
	for(i=0; i<h; i++) {
		for(j=0; j<w; j++) {

            // conversion (i,j) -> (x,y)
			a = a0 + i * da;
			b = b0 + j * db;

			iter = mandelbrot_scalar(a, b, max_iter);

			M[i+j] = iter;
		}
	}
}

// -----------------------------------------------------------------------------------------------------------
void calc_mandelbrot_simd(vuint32 **M, size_t h, size_t w, float a0, float a1, float b0, float b1, size_t max_iter)
// -----------------------------------------------------------------------------------------------------------
{

	
	size_t i, j;

	float da, db;
	float sa, sb;
	vfloat32 a, b;
	vuint32 iter;

	da = (a1-a0)/w;
	db = (b1-b0)/h;
	
	#pragma omp parallel for
	for(i=0; i<h; i++) {
		for(j=0; j<w/4; j++) {

            // conversion (i,j) -> (x,y)
			sa = a0 + i * da;
			sb = b0 + j * db * 4;

			a = _mm_setr_ps(sa, sa+da, sa+2*da, sa+3*da);
			b = _mm_set1_ps(sb);

			iter = mandelbrot_simd(a, b, max_iter);
			M[i][j] = iter;
		}
	}
   }

// Prints an simd float vector
void simd_display_f32(vfloat32 v)
{
	float out[4] ;
	_mm_store_ps( out, v);
	std::cout << "simd :"  << out[0] << " " << out[1] << " " << out[2] << " " << out[3] << std::endl;
}

// Prints an simd int vector
void simd_display_i32(__m128i var)
{
	int out[4] __attribute__((aligned(16))) ;

	__m128i* po = (__m128i*) &out[0] ;

	_mm_store_si128(po, var);
	printf(" SIMD : %i %i %i %i  \n",
		out[3], out[2], out[1], out[0]);
}

// Prints 4 values of a vector starting from i;
template<typename T>
void vec_display(std::vector<T> & vec , size_t i )
{
	if( i + 3 < vec.size())
		{std::cout << " vector :"  << vec[i] << " " << vec[i+1] << " " << vec[i+2] << " " << vec[i+3] << std::endl;}
}

int main()
{
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<float> dis(0, 255);

	size_t max_iter = 20;
	size_t array_size = 800;
	size_t vector_size = array_size*4;

	vfloat32 *vX1, *vX2, *vY , *vY1 , *vY2;
	std::vector<float> vec1(vector_size) , vec2(vector_size) , vecy(vector_size , 0.) , vecy1(vector_size,0.)
	, vecy2(vector_size, 0.);

	// SIMD vectors must be 16 bits aligned
	vX1 =(vfloat32*) _mm_malloc ((size_t) (array_size*sizeof(vfloat32)), 16);
	vX2 =(vfloat32*) _mm_malloc ((size_t) (array_size*sizeof(vfloat32)), 16);
	vY  =(vfloat32*) _mm_malloc ((size_t) (array_size*sizeof(vfloat32)), 16);
	vY1 =(vfloat32*) _mm_malloc ((size_t) (array_size*sizeof(vfloat32)), 16);
	vY2 =(vfloat32*) _mm_malloc ((size_t) (array_size*sizeof(vfloat32)), 16);

	vfloat32 vy = _mm_set_ps(0,0,0,0);

	int j = 0;

	// Initialize vectors and simd arrays
	for(size_t i = 0 ; i < array_size ; ++i)
	{
		float r1 = dis(gen) , r2 = dis(gen) , r3 = dis(gen) , r4 = dis(gen);
		float r5 = dis(gen) , r6 = dis(gen) , r7 = dis(gen) , r8 = dis(gen);

		vec1[j] = r1; vec1[j+1] = r2 ; vec1[j+2] = r3 ; vec1[j+3] = r4;
		vec2[j] = r5; vec2[j+1] = r6 ; vec2[j+2] = r7 ; vec2[j+3] = r8;

		vfloat32 vx1 = _mm_set_ps(r4 , r3 , r2 , r1  );
		vfloat32 vx2 = _mm_set_ps(r8 , r7 , r6 , r5  );

		_mm_store_ps((float*) &vX1[i], vx1);
		_mm_store_ps((float*) &vX2[i], vx2);
		_mm_store_ps((float*) &vY[i], vy);
		_mm_store_ps((float*) &vY1[i], vy);
		_mm_store_ps((float*) &vY2[i], vy);

		j +=4;
	}

	// test pour l'addition de vectors
	{
		auto start = std::chrono::steady_clock::now();
		vectoradd_simd(vX1,vX2,vY,array_size);
		auto end = std::chrono::steady_clock::now();
		std::chrono::duration<double> diff = end-start;
		// std::cout << "vector addition time with simd: " << diff.count() << " s" << std::endl;

		start = std::chrono::steady_clock::now();
		std::transform( vec1.begin() , vec1.end() , vec2.begin() , vecy.begin() , std::plus<float>());

		end = std::chrono::steady_clock::now();
		std::chrono::duration<double> diff1 = end-start;
		// std::cout << "vector addition time without simd: " << diff1.count() << " s" << std::endl;

		j = 0;
		bool is_valid = true;
		for(size_t i = 0 ; i < array_size ; ++i)
		{
			float out[4] ;
			_mm_store_ps(out , vY[i]);

			if ( out[0] == vecy[j] && out[1] == vecy[j+1] && out[2] == vecy[j+2] && out[3] == vecy[j+3])
				{ j += 4;}
			else
			{
				is_valid = false;
				break;
			}
		}

		if(is_valid)
		{
			std::cout << "l'addition de vecteurs en simd est correcte" << std::endl;
			std::cout << "speedup obtained for vector addition with simd : " << diff1.count() / diff.count() << std::endl;
		}
		else
		{
			std::cout << " l'addition de vecteurs end simd est incorrecte" << std::endl;
		}

		std::cout << "\n";
	}

	// test pour le dot product
	{
		auto start = std::chrono::steady_clock::now();
		vfloat32 sres = vectordot_simd(vX1 , vX2 , array_size);
		auto end = std::chrono::steady_clock::now();
		std::chrono::duration<double> diff = end-start;
		// std::cout << "dot product time with simd: " << diff.count() << " s" << std::endl;

		start = std::chrono::steady_clock::now();
		float res = std::inner_product( vec1.begin() , vec1.end() , vec2.begin() , 0. );
		end = std::chrono::steady_clock::now();
		std::chrono::duration<double> diff1 = end-start;
		// std::cout << "dot product time without simd: " << diff1.count() << " s" << std::endl;

		float out[4] ;
		_mm_store_ps( out , sres);

		if(  std::abs(out[0] - res ) < 0.01f )
		{
			std::cout << "le produit de vecteurs en simd est correct" << std::endl;
			std::cout << "speedup obtained for dot product with simd : " << diff1.count() / diff.count() << std::endl;
		}
		else {std::cout << "le produit de vecteurs en simd est incorrect : " << out[0] << "  " << res << std::endl;}

		std::cout << "\n";
	}

	// test for 1D filtre with rotation without border check
	{
		auto start = std::chrono::steady_clock::now();
		float divide = 1./3. ;
		for(std::size_t i = 1 ; i < vector_size-1 ; ++i)
		{
			vecy1[i] = divide * ( vec1[i-1] + vec1[i] + vec1[i+1] );
		}
		auto end = std::chrono::steady_clock::now();
		std::chrono::duration<double> diff1 = end-start;;

		start = std::chrono::steady_clock::now();
		vectoravg3_simd(vX1 , vY1 , array_size);
		end = std::chrono::steady_clock::now();
		std::chrono::duration<double> diff = end-start;

		j = 4;
		bool is_valid = true;

		for(size_t i = 1 ; i < array_size-1 ; ++i)
		{
			float out[4] ;
			_mm_store_ps(out , vY1[i]);

			if ( is_valid == true && out[0] == vecy1[j] && out[1] == vecy1[j+1] && out[2] == vecy1[j+2] && out[3] == vecy1[j+3])
				{ j += 4;}
			else
			{
				is_valid = false;
				break;
			}
		}

		if(is_valid)
		{
			std::cout << "la filtre moyenneur en simd est correct" << std::endl;
			std::cout << "speedup obtained for average filter with simd : " << diff1.count() / diff.count() << std::endl;
		}
		else
		{
			std::cout << "la filtre moyenneur en simd est incorrect" << std::endl;
		}

		std::cout << "\n";
	}

	bool valid_mandel = false;
	// test for mandelbrot
	{
		std::vector<float> mandel_test(4,0);
		std::vector<float> mandel_test1(4,0);
		std::vector<size_t> indx(4,0);
		vfloat32 mdt = _mm_set1_ps(0);
		vfloat32 mdt1 = _mm_set1_ps(0);

		mandel_test[0] = -0.70;
		mandel_test[1] = -0.80;
		mandel_test[2] = -0.90;
		mandel_test[3] = -1.00;

		mandel_test1[0] = +0.10;
		mandel_test1[1] = +0.30;
		mandel_test1[2] = +0.30;
		mandel_test1[3] = +0.40;

		mdt  = _mm_setr_ps(-1.00, -0.90, -0.80, -0.70);
		mdt1 = _mm_setr_ps(+0.40, +0.30, +0.30, +0.10);

		auto start = std::chrono::steady_clock::now();
		for(std::size_t i = 0 ; i < 4 ; ++i )
		{

			indx[i] = mandelbrot_scalar(mandel_test[i] , mandel_test1[i] , max_iter );
		}
		auto end = std::chrono::steady_clock::now();
		std::chrono::duration<double> diff1 = end-start;;

		start = std::chrono::steady_clock::now();
		vuint32 res_mandel = mandelbrot_simd(mdt, mdt1 , max_iter);
		end = std::chrono::steady_clock::now();
		std::chrono::duration<double> diff = end-start;

		unsigned int out[4] __attribute__((aligned(16))) ;

		__m128i* po = (__m128i*) &out[0] ;

		_mm_store_si128(po, res_mandel);

		bool v1 = false , v2 = false;

		if( indx[0] == 20 && indx[1] == 8 && indx[2] == 10 && indx[3] == 6 )
		{
			v1 = true;
			std::cout << "la fonction mandelbrot en scalaire est correcte" << std::endl;
		}
		else
		{
			std::cout << "la fonction mandelbrot en scalaire est incorrecte" << std::endl;
			std::cout << "le bon résultat est : 20 8 10 6 \n" << "vous avez obtenu : ";
			vec_display(indx,0);
		}


		if( out[3] == 20 && out[2] == 8 && out[1] == 10 && out[0] == 6 )
		{
			v2 = true;
			std::cout << "la fonction mandelbrot en SIMD est correcte" << std::endl;
		}
		else
		{
			std::cout << "la fonction mandelbrot en SIMD est incorrecte" << std::endl;
			std::cout << "le bon résultat est 20 8 10 6 \n" << "vous avez obtenu :  ";
			simd_display_i32(res_mandel);
		}


		if ( v1 && v2 )
		{
			std::cout << "speedup obtained for mandelbrot : " << diff1.count() / diff.count() << std::endl;
			valid_mandel = true;
		}
	}

	// test for mandelbrot function

	{
		if(valid_mandel)
		{

			std::cout << "\n-----------------------------" << std::endl;
			std::cout << "------ benchmandelbrot ------" << std::endl;
			std::cout << "-----------------------------\n" << std::endl;

			size_t h = SIZE , w = SIZE ;
			std::vector<size_t> indx(h*w,0);
			vfloat32 mdt = _mm_set1_ps(0);
			vfloat32 mdt1 = _mm_set1_ps(0);

			float a0 = -1.5 , a1 = +0.5;
			float b0 = -1.0 , b1 = +1.0;

			float avg_cycles_vec = 0;
			float avg_time_vec  = 0;

			size_t num_iter = 200;

			for(size_t i =0 ; i < num_iter ; ++i)
			{
				auto start = std::chrono::steady_clock::now();
				auto cycles_s = rdtsc();
				calc_mandelbrot_scalar( indx , h , w , a0 , a1 , b0 , b1  , max_iter );
				auto cycles_e = rdtsc();
				auto end = std::chrono::steady_clock::now();
				std::chrono::duration<double> diff1 = end-start;

				avg_time_vec += diff1.count() ;

				avg_cycles_vec += cycles_e - cycles_s;
			}

			avg_time_vec /= num_iter ;
			avg_cycles_vec /= num_iter ;

			std::cout << " mandelbrot vector time : " << avg_time_vec << std::endl;
			std::cout << " mandelbrot vector cycles time : " << avg_cycles_vec << std::endl;

			vuint32 **Simd_indx = (vuint32**)_mm_malloc ((size_t)( h*sizeof(vuint32*)), 16);
			if (Simd_indx)
			{
				for (size_t i = 0; i < w; i++)
				{
					Simd_indx[i] = (vuint32*) _mm_malloc ((size_t) (w*sizeof(vuint32)), 16);
				}
			}

			float avg_cycles_simd = 0;
			float avg_time_simd  = 0;

			for(size_t i = 0 ; i < num_iter ; ++i)
			{
				auto start = std::chrono::steady_clock::now();
				auto cycles_s = rdtsc();
				calc_mandelbrot_simd( Simd_indx , h , w , a0 , a1 , b0 , b1  , max_iter );
				auto cycles_e = rdtsc();
				auto end = std::chrono::steady_clock::now();
				std::chrono::duration<double> diff = end-start;

				avg_time_simd += diff.count() ;
				avg_cycles_simd += cycles_e - cycles_s;
			}

			avg_time_simd /= num_iter ;
			avg_cycles_simd /= num_iter ;

			std::cout << " mandelbrot SIMD time : " << avg_time_simd << std::endl;
			std::cout << " mandelbrot SIMD cycles time : " << avg_cycles_simd << std::endl;

			std::cout << "speedup obtained for mandelbrot : " << avg_time_vec / avg_time_simd << std::endl;
			std::cout << "speedup in cycles obtained for mandelbrot : " <<  avg_cycles_vec / avg_cycles_simd << std::endl;
		}

	}


	_mm_free(vX1);
	_mm_free(vX2);
	_mm_free(vY);
	_mm_free(vY1);
	_mm_free(vY2);


}
