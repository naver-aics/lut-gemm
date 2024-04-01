#include "custom_random.h"

void random_seed(){
  time_t t;
  srand((unsigned int)time(&t));
}
bool rand_bool(){
  return rand()>(RAND_MAX/2);
}
double rand_fp64(double max){
  double sign[] = {-1.0,1.0};
  return (double)sign[rand_bool()]*rand()/RAND_MAX*rand()/RAND_MAX*max;
}

float rand_fp32(float max){
  return rand_fp64()*max;
}