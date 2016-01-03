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

// Redefinition of a 128 bits simd vector
typedef __m128 vfloat32;

#define vect_left1(v0,v1)

#define vect_right1(v0,v1)


// -------------------------------------------------------------------
void vectoradd_simd( vfloat32 *vX1, vfloat32 *vX2, vfloat32 *vY , size_t size)
// -------------------------------------------------------------------
{
    vfloat32 x1, x2, y;

    for(size_t i=0; i < size; ++i)
    {
    	// Remplir le corps de la fonction d'addition de vecteurs
      x1 = _mm_load_ps((float*)(&vX1));
      x2 = _mm_load_ps((float*)(&vX2));
      y = _mm_add_ps(x1, x2);
      _mm_store_ps((float*)&vY[i], y);
    }
}

// ---------------------------------------------------------
vfloat32 vectordot_simd(vfloat32 *vX1, vfloat32 *vX2, size_t size)
// ---------------------------------------------------------
{
  vfloat32 x1, x2, p, s;

  for(size_t i=0; i<size; ++i)
  {
	// Remplir le corps de la fonction produit scalaire
  }

  
  return s; // attention il faut retourner un registre SIMD et non un scalaire
}

// ----------------------------------------------------
void vectoravg3_simd(vfloat32 *vX, vfloat32 *vY , size_t size)
// ----------------------------------------------------
{
  vfloat32 x1,x2,x3;

  // on va ignorer les bords
  for(size_t i=1 ; i<size-1 ; ++i)
  {
  	// Remplir le corps de la fonction de moyennage 1D

    x3 = _mm_shuffle_ps(x1, x2, _MM_SHUFFLE(2,3,0,1));
  }
}

// --------------------------------------------------------
void vectoravg3_rot_simd(vfloat32 *vX, vfloat32 *vY , size_t size)
// --------------------------------------------------------
{
  vfloat32 x1,x2,x3,x4;


  for(size_t i=1 ;i<size-1;i++)
  {
  	// Remplir le corps de la fonction de moyennage 1D rot
  }
}


// Prints an simd vector
void simd_display(vfloat32 v)
{
	float out[4] ;
	_mm_store_ps( out, v);
	std::cout << "simd :"  << out[3] << " " << out[2] << " " << out[1] << " " << out[0] << std::endl;
}

// Prints an simd vector
void vec_display(std::vector<float> & vec , size_t i )
{
	if( i + 3 < vec.size())
	{std::cout << "vec :"  << vec[i] << " " << vec[i+1] << " " << vec[i+2] << " " << vec[i+3] << std::endl;}
}

int main()
{
	// Generation of random number
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<float> dis(1, 5);

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

		if(is_valid) {std::cout << "l'addition de vecteurs en simd est correcte" << std::endl;
					  std::cout << "speedup obtained for vector addition with simd : " << diff1.count() / diff.count() << std::endl;
					 }
		else {std::cout << " l'addition de vecteurs en simd est incorrecte" << std::endl;}

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
		else
		{
			std::cout << "le produit de vecteurs en simd est incorrect :\n"
			<< "valeur en simd : " << out[0]
			<< "\nvaleur correcte :" << res << std::endl;}

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

		if(is_valid) {std::cout << "la filtre moyenneur en simd est correct" << std::endl;
					  std::cout << "speedup obtained for average filter with simd : " << diff1.count() / diff.count() << std::endl;
					 }
		else {std::cout << "la filtre moyenneur en simd est incorrect" << std::endl;}

		std::cout << "\n";
	}

	// test for 1D filter with rotation without border check
	{
		auto start = std::chrono::steady_clock::now();
		float divide = 1./3. ;
		for(std::size_t i = 1 ; i < vector_size-1 ; ++i)
		{
			vecy2[i] = divide * ( vec1[i-1] + vec1[i] + vec1[i+1] );
		}
		auto end = std::chrono::steady_clock::now();
		std::chrono::duration<double> diff1 = end-start;;

		start = std::chrono::steady_clock::now();
		vectoravg3_rot_simd(vX1 , vY2 , array_size);
		end = std::chrono::steady_clock::now();
		std::chrono::duration<double> diff = end-start;

		j = 4;
		bool is_valid = true;

		for(size_t i = 1 ; i < array_size-1 ; ++i)
		{
			float out[4] ;
			_mm_store_ps(out , vY2[i]);

			if ( is_valid == true && out[0] == vecy2[j] && out[1] == vecy2[j+1] && out[2] == vecy2[j+2] && out[3] == vecy2[j+3])
			{ j += 4;}
			else
			{
				is_valid = false;
				break;
			}
		}

		if(is_valid) {std::cout << "la filtre moyenneur avec rotation en simd est correct" << std::endl;
					  std::cout << "speedup obtained for average filter wiht rotation in simd : " << diff1.count() / diff.count() << std::endl;
					 }
		else {std::cout << "la filtre moyenneur avec rotation en simd est incorrect" << std::endl;}
	}

	 _mm_free(vX1);
	 _mm_free(vX2);
	 _mm_free(vY);
	 _mm_free(vY1);
	 _mm_free(vY2);




}
