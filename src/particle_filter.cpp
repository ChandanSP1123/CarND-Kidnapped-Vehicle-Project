/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include <float.h>

#include "helper_functions.h"

using std::string;
using std::vector;
using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[])
{

  // setting the number of particle
  num_particles = 100; // 
  

  // normal distribution creation

  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);
  default_random_engine gen;
  for (int i = 0; i < num_particles; i++)
  {
    Particle particle;
    particle.id = i;
    particle.x = dist_x(gen);
    particle.y = dist_y(gen);
    particle.theta = dist_theta(gen);
    particle.weight = 1.0;

    particles.push_back(particle);
  }
  is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[],
                                double velocity, double yaw_rate)
{


   //  standard deviations

    normal_distribution<double> d_x(0, std_pos[0]);
    normal_distribution<double> d_y(0, std_pos[1]);
    normal_distribution<double> d_theta(0, std_pos[2]);
    default_random_engine gen;  
  
  
  for (int i = 0; i < num_particles; i++)
  {
    
    if (fabs(yaw_rate) < 0.00001)
    {
      particles[i].x = particles[i].x + velocity * cos(particles[i].theta) * delta_t;
      particles[i].y = particles[i].y + velocity * sin(particles[i].theta) * delta_t;
   
    }
    else
    {
      particles[i].x = particles[i].x + (velocity / yaw_rate) * (sin(particles[i].theta + (yaw_rate * delta_t)) - sin(particles[i].theta));
      particles[i].y = particles[i].y + (velocity / yaw_rate) * (cos(particles[i].theta) - cos(particles[i].theta + (yaw_rate * delta_t)));
      particles[i].theta = particles[i].theta + yaw_rate * delta_t;
    }


    //adding noise
    particles[i].x = particles[i].x + d_x(gen);
    particles[i].y =  particles[i].y + d_y(gen);
    particles[i].theta = particles[i].theta + d_theta(gen);
  }
  
  
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted,
                                     vector<LandmarkObs> &observations)
{
 
  
  for (unsigned int i = 0; i < observations.size(); i++)
  {
    double minimumDistance =  DBL_MAX;
    int id = -1;
    for (unsigned int j = 0; j < predicted.size(); j++)
    {
      double distance = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);
      if (minimumDistance > distance)
      {
        minimumDistance = distance;
       id = predicted[j].id;
      }
    }
    observations[i].id =id;
  }
 
  }

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const vector<LandmarkObs> &observations,
                                   const Map &map_landmarks)
{
 

  for (int i = 0; i < num_particles; i++)
  {
     particles[i].weight = 1.0 ;

    // landmarks near by shall be collected
    vector<LandmarkObs> predictions;
    for (unsigned int j = 0; j < map_landmarks.landmark_list.size(); j++)
    {
      Map::single_landmark_s landmark = map_landmarks.landmark_list[j];
      double distance = dist(landmark.x_f, landmark.y_f, particles[i].x, particles[i].y);
      if (distance < sensor_range)
      {
        predictions.push_back(LandmarkObs{landmark.id_i, landmark.x_f, landmark.y_f});
      }
    }
      //convert the coordinates to Map
      vector<LandmarkObs> ObservationCoordinate;
    for(const auto& obs: observations)
      {
        LandmarkObs Local_copy;
        Local_copy.id = obs.id;
        Local_copy.x = obs.x * cos(particles[i].theta) - obs.y * sin(particles[i].theta) + particles[i].x;
        Local_copy.y = obs.x * sin(particles[i].theta) + obs.y * cos(particles[i].theta) + particles[i].y;
        ObservationCoordinate.push_back(Local_copy);
      }
    //landmark index  locating 
    dataAssociation(predictions,ObservationCoordinate);
  // calculation of particle weight 
    for(const auto &obs_cor:ObservationCoordinate){

      Map::single_landmark_s landmark = map_landmarks.landmark_list.at(obs_cor.id-1);
      double temp_x = pow(obs_cor.x - landmark.x_f, 2) / (2 * pow(std_landmark[0], 2));
      double temp_y = pow(obs_cor.y - landmark.y_f, 2) / (2 * pow(std_landmark[1], 2));
      double w = exp(-(temp_x + temp_y)) / (2 * M_PI * std_landmark[0] * std_landmark[1]);
      particles[i].weight *= w;
    }
  weights.push_back(particles[i].weight);
  }

  
}

void ParticleFilter::resample()
{
 
  vector<Particle> new_particles;
  new_particles.resize(num_particles);
  std::random_device random;  
  std::mt19937 gen(random());
  std::discrete_distribution<> dist(weights.begin(),weights.end());
  for(int i =0 ; i<num_particles;i++){
     int ind = dist(gen);
  new_particles[i] = particles[ind];
  }
  particles = new_particles;
  weights.clear();

}

void  ParticleFilter::SetAssociations(Particle &particle,
                                     const vector<int> &associations,
                                     const vector<double> &sense_x,
                                     const vector<double> &sense_y)
{
  // particle: the particle to which assign each listed association,
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates

  particle.associations.clear();
  particle.sense_x.clear();
  particle.sense_y.clear();

  particle.associations = associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;

 
}

string ParticleFilter::getAssociations(Particle best)
{
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1); // get rid of the trailing space
  return s;
  
}

string ParticleFilter::getSenseCoord(Particle best, string coord)
{
  vector<double> v;

  if (coord == "X")
  {
    v = best.sense_x;
  }
  else
  {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1); // get rid of the trailing space
  return s;
  
}
