/**
 * @file 	fsi2.cpp
 * @brief 	This is the benchmark test of fluid-structure interaction.
 * @details We consider a flow-induced vibration of an elastic beam behind a cylinder in 2D.
 *			The case can be found in Chi Zhang, Massoud Rezavand, Xiangyu Hu,
 *			Dual-criteria time stepping for weakly compressible smoothed particle hydrodynamics.
 *			Journal of Computation Physics 404 (2020) 109135.
 * @author 	Xiangyu Hu, Chi Zhang and Luhui Han
 */
#include "sphinxsys.h"

#include "fsi2.h" //	case file to setup the test case
using namespace SPH;

int main(int ac, char *av[])
{
	//----------------------------------------------------------------------
	//	Build up the environment of a SPHSystem with global controls.
	//----------------------------------------------------------------------
	SPHSystem sph_system(system_domain_bounds, resolution_ref);
	/** Tag for run particle relaxation for the initial body fitted distribution. */
	sph_system.run_particle_relaxation_ = false;
	/** Tag for computation start with relaxed body fitted particles distribution. */
	sph_system.reload_particles_ = true;
	/** Tag for computation from restart files. 0: start with initial condition. */
	sph_system.restart_step_ = 0;
// handle command line arguments
#ifdef BOOST_AVAILABLE
	sph_system.handleCommandlineOptions(ac, av);
#endif
	IOEnvironment io_environment(sph_system);
	//----------------------------------------------------------------------
	//	Creating body, materials and particles.
	//----------------------------------------------------------------------
	FluidBody water_block(sph_system, makeShared<WaterBlock>("WaterBody"));
	water_block.defineParticlesAndMaterial<FluidParticles, WeaklyCompressibleFluid>(rho0_f, c_f, mu_f);
	water_block.generateParticles<ParticleGeneratorLattice>();

	SolidBody wall_boundary(sph_system, makeShared<WallBoundary>("Wall"));
	wall_boundary.defineParticlesAndMaterial<SolidParticles, Solid>();
	wall_boundary.generateParticles<ParticleGeneratorLattice>();

	SolidBody insert_body(sph_system, makeShared<Insert>("InsertedBody"));
	insert_body.sph_adaptation_->resetAdaptationRatios(1.15, 2.0);
	insert_body.defineBodyLevelSetShape()->writeLevelSet(insert_body);
	insert_body.defineParticlesAndMaterial<ElasticSolidParticles, LinearElasticSolid>(rho0_s, Youngs_modulus, poisson);
	(!sph_system.run_particle_relaxation_ && sph_system.reload_particles_)
		? insert_body.generateParticles<ParticleGeneratorReload>(io_environment, insert_body.getName())
		: insert_body.generateParticles<ParticleGeneratorLattice>();

	ObserverBody beam_observer(sph_system, "BeamObserver");
	beam_observer.generateParticles<ObserverParticleGenerator>(beam_observation_location);
	ObserverBody fluid_observer(sph_system, "FluidObserver");
	fluid_observer.generateParticles<FluidObserverParticleGenerator>();
	//----------------------------------------------------------------------
	//	Define body relation map.
	//	The contact map gives the topological connections between the bodies.
	//	Basically the the range of bodies to build neighbor particle lists.
	//----------------------------------------------------------------------
	BodyRelationInner insert_body_inner(insert_body);
	ComplexBodyRelation water_block_complex(water_block, RealBodyVector{&wall_boundary, &insert_body});
	BodyRelationContact insert_body_contact(insert_body, {&water_block});
	BodyRelationContact beam_observer_contact(beam_observer, {&insert_body});
	BodyRelationContact fluid_observer_contact(fluid_observer, {&water_block});
	//----------------------------------------------------------------------
	//	Run particle relaxation for body-fitted distribution if chosen.
	//----------------------------------------------------------------------
	if (sph_system.run_particle_relaxation_)
	{
		//----------------------------------------------------------------------
		//	Methods used for particle relaxation.
		//----------------------------------------------------------------------
		/** Random reset the insert body particle position. */
		SimpleDynamics<RandomizeParticlePosition> random_insert_body_particles(insert_body);
		/** Write the body state to Vtp file. */
		BodyStatesRecordingToVtp write_insert_body_to_vtp(io_environment, {&insert_body});
		/** Write the particle reload files. */
		ReloadParticleIO write_particle_reload_files(io_environment, {&insert_body});
		/** A  Physics relaxation step. */
		relax_dynamics::RelaxationStepInner relaxation_step_inner(insert_body_inner);
		//----------------------------------------------------------------------
		//	Particle relaxation starts here.
		//----------------------------------------------------------------------
		random_insert_body_particles.parallel_exec(0.25);
		relaxation_step_inner.surface_bounding_.parallel_exec();
		write_insert_body_to_vtp.writeToFile(0);
		//----------------------------------------------------------------------
		//	Relax particles of the insert body.
		//----------------------------------------------------------------------
		int ite_p = 0;
		while (ite_p < 1000)
		{
			relaxation_step_inner.parallel_exec();
			ite_p += 1;
			if (ite_p % 200 == 0)
			{
				std::cout << std::fixed << std::setprecision(9) << "Relaxation steps for the inserted body N = " << ite_p << "\n";
				write_insert_body_to_vtp.writeToFile(ite_p);
			}
		}
		std::cout << "The physics relaxation process of inserted body finish !" << std::endl;
		/** Output results. */
		write_particle_reload_files.writeToFile(0);
		return 0;
	}
	//----------------------------------------------------------------------
	//	Define the main numerical methods used in the simulation.
	//	Note that there may be data dependence on the constructors of these methods.
	//----------------------------------------------------------------------
	/** Initialize particle acceleration. */
	SimpleDynamics<TimeStepInitialization> initialize_a_fluid_step(water_block);
	/** Evaluation of density by summation approach. */
	fluid_dynamics::DensitySummationComplex update_density_by_summation(water_block_complex);
	/** Time step size without considering sound wave speed. */
	ReduceDynamics<fluid_dynamics::AdvectionTimeStepSize> get_fluid_advection_time_step_size(water_block, U_f);
	/** Time step size with considering sound wave speed. */
	ReduceDynamics<fluid_dynamics::AcousticTimeStepSize> get_fluid_time_step_size(water_block);
	/** Pressure relaxation using verlet time stepping. */
	/** Here, we do not use Riemann solver for pressure as the flow is viscous. */
	fluid_dynamics::PressureRelaxationWithWall pressure_relaxation(water_block_complex);
	fluid_dynamics::DensityRelaxationRiemannWithWall density_relaxation(water_block_complex);
	/** Computing viscous acceleration. */
	NewInteractionDynamics<fluid_dynamics::ViscousAccelerationWithWall> viscous_acceleration(water_block_complex);
	/** Impose transport velocity. */
	NewInteractionDynamics<fluid_dynamics::TransportVelocityCorrectionComplex> transport_velocity_correction(water_block_complex);
	/** viscous acceleration and transport velocity correction can be combined because they are independent dynamics. */
	NewInteractionDynamics<CombinedLocalInteractionDynamics<
		fluid_dynamics::ViscousAccelerationWithWall,
		fluid_dynamics::TransportVelocityCorrectionComplex>> 
		viscous_acceleration_and_transport_correction(water_block_complex);
	//	CombinedInteractionDynamics viscous_acceleration_and_transport_correction(viscous_acceleration, transport_velocity_correction);
	/** Computing vorticity in the flow. */
	NewInteractionDynamics<fluid_dynamics::VorticityInner> compute_vorticity(water_block_complex.inner_relation_);
	/** Inflow boundary condition. */
	BodyAlignedBoxByCell inflow_buffer(
		water_block, makeShared<AlignedBoxShape>(Transform2d(Vec2d(buffer_translation)), buffer_halfsize));
	SimpleDynamics<ParabolicInflow, BodyAlignedBoxByCell> parabolic_inflow(inflow_buffer);
	/** Periodic BCs in x direction. */
	PeriodicConditionUsingCellLinkedList periodic_condition(water_block, water_block.getBodyShapeBounds(), xAxis);
	//----------------------------------------------------------------------
	//	Algorithms of FSI.
	//----------------------------------------------------------------------
	SimpleDynamics<NormalDirectionFromBodyShape> wall_boundary_normal_direction(wall_boundary);
	SimpleDynamics<NormalDirectionFromBodyShape> insert_body_normal_direction(insert_body);
	/** Corrected configuration for the elastic insert body. */
	NewInteractionDynamics<solid_dynamics::CorrectConfiguration> insert_body_corrected_configuration(insert_body_inner);
	/** Compute the force exerted on solid body due to fluid pressure and viscosity. */
	solid_dynamics::FluidForceOnSolidUpdate fluid_force_on_solid_update(insert_body_contact);
	/** Compute the average velocity of the insert body. */
	solid_dynamics::AverageVelocityAndAcceleration average_velocity_and_acceleration(insert_body);
	//----------------------------------------------------------------------
	//	Algorithms of solid dynamics.
	//----------------------------------------------------------------------
	/** Compute time step size of elastic solid. */
	ReduceDynamics<solid_dynamics::AcousticTimeStepSize> insert_body_computing_time_step_size(insert_body);
	/** Stress relaxation for the inserted body. */
	solid_dynamics::StressRelaxationFirstHalf insert_body_stress_relaxation_first_half(insert_body_inner);
	solid_dynamics::StressRelaxationSecondHalf insert_body_stress_relaxation_second_half(insert_body_inner);
	/** Constrain region of the inserted body. */
	BodyRegionByParticle beam_base(insert_body, makeShared<MultiPolygonShape>(createBeamBaseShape()));
	SimpleDynamics<solid_dynamics::FixConstraint, BodyRegionByParticle> constraint_beam_base(beam_base);
	/** Update norm .*/
	SimpleDynamics<solid_dynamics::UpdateElasticNormalDirection> insert_body_update_normal(insert_body);
	//----------------------------------------------------------------------
	//	Define the methods for I/O operations and observations of the simulation.
	//----------------------------------------------------------------------
	BodyStatesRecordingToVtp write_real_body_states(io_environment, sph_system.real_bodies_);
	RestartIO restart_io(io_environment, sph_system.real_bodies_);
	RegressionTestTimeAveraged<BodyReducedQuantityRecording<ReduceDynamics<solid_dynamics::TotalViscousForceOnSolid>>>
		write_total_viscous_force_on_insert_body(io_environment, insert_body);
	RegressionTestDynamicTimeWarping<ObservedQuantityRecording<Vecd>>
		write_beam_tip_displacement("Position", io_environment, beam_observer_contact);
	ObservedQuantityRecording<Vecd>
		write_fluid_velocity("Velocity", io_environment, fluid_observer_contact);
	//----------------------------------------------------------------------
	//	Prepare the simulation with cell linked list, configuration
	//	and case specified initial condition if necessary.
	//----------------------------------------------------------------------
	/** initialize cell linked lists for all bodies. */
	sph_system.initializeSystemCellLinkedLists();
	/** periodic condition applied after the mesh cell linked list build up
	 * but before the configuration build up. */
	periodic_condition.update_cell_linked_list_.parallel_exec();
	/** initialize configurations for all bodies. */
	sph_system.initializeSystemConfigurations();
	/** computing surface normal direction for the wall. */
	wall_boundary_normal_direction.parallel_exec();
	/** computing surface normal direction for the insert body. */
	insert_body_normal_direction.parallel_exec();
	/** computing linear reproducing configuration for the insert body. */
	insert_body_corrected_configuration.parallel_exec();
	//----------------------------------------------------------------------
	//	Load restart file if necessary.
	//----------------------------------------------------------------------
	if (sph_system.restart_step_ != 0)
	{
		GlobalStaticVariables::physical_time_ = restart_io.readRestartFiles(sph_system.restart_step_);
		insert_body.updateCellLinkedList();
		water_block.updateCellLinkedList();
		periodic_condition.update_cell_linked_list_.parallel_exec();
		/** one need update configuration after periodic condition. */
		water_block_complex.updateConfiguration();
		insert_body_contact.updateConfiguration();
		insert_body_update_normal.parallel_exec();
	}
	//----------------------------------------------------------------------
	//	Setup computing and initial conditions.
	//----------------------------------------------------------------------
	size_t number_of_iterations = sph_system.restart_step_;
	int screen_output_interval = 100;
	int restart_output_interval = screen_output_interval * 10;
	Real end_time = 200.0;
	Real output_interval = end_time / 200.0;
	//----------------------------------------------------------------------
	//	Statistics for CPU time
	//----------------------------------------------------------------------
	tick_count t1 = tick_count::now();
	tick_count::interval_t interval;
	//----------------------------------------------------------------------
	//	First output before the main loop.
	//----------------------------------------------------------------------
	write_real_body_states.writeToFile();
	write_beam_tip_displacement.writeToFile(number_of_iterations);
	//----------------------------------------------------------------------
	//	Main loop starts here.
	//----------------------------------------------------------------------
	while (GlobalStaticVariables::physical_time_ < end_time)
	{
		Real integration_time = 0.0;
		/** Integrate time (loop) until the next output time. */
		while (integration_time < output_interval)
		{
			initialize_a_fluid_step.parallel_exec();
			Real Dt = get_fluid_advection_time_step_size.parallel_exec();
			update_density_by_summation.parallel_exec();
			viscous_acceleration_and_transport_correction.parallel_exec(Dt);

			/** FSI for viscous force. */
			fluid_force_on_solid_update.viscous_force_.parallel_exec();
			/** Update normal direction on elastic body.*/
			insert_body_update_normal.parallel_exec();
			size_t inner_ite_dt = 0;
			size_t inner_ite_dt_s = 0;
			Real relaxation_time = 0.0;
			while (relaxation_time < Dt)
			{
				Real dt = SMIN(get_fluid_time_step_size.parallel_exec(), Dt);
				/** Fluid pressure relaxation */
				pressure_relaxation.parallel_exec(dt);
				/** FSI for pressure force. */
				fluid_force_on_solid_update.parallel_exec();
				/** Fluid density relaxation */
				density_relaxation.parallel_exec(dt);

				/** Solid dynamics. */
				inner_ite_dt_s = 0;
				Real dt_s_sum = 0.0;
				average_velocity_and_acceleration.initialize_displacement_.parallel_exec();
				while (dt_s_sum < dt)
				{
					Real dt_s = SMIN(insert_body_computing_time_step_size.parallel_exec(), dt - dt_s_sum);
					insert_body_stress_relaxation_first_half.parallel_exec(dt_s);
					constraint_beam_base.parallel_exec();
					insert_body_stress_relaxation_second_half.parallel_exec(dt_s);
					dt_s_sum += dt_s;
					inner_ite_dt_s++;
				}
				average_velocity_and_acceleration.update_averages_.parallel_exec(dt);

				relaxation_time += dt;
				integration_time += dt;
				GlobalStaticVariables::physical_time_ += dt;
				parabolic_inflow.parallel_exec();
				inner_ite_dt++;
			}

			if (number_of_iterations % screen_output_interval == 0)
			{
				std::cout << std::fixed << std::setprecision(9) << "N=" << number_of_iterations << "	Time = "
						  << GlobalStaticVariables::physical_time_
						  << "	Dt = " << Dt << "	Dt / dt = " << inner_ite_dt << "	dt / dt_s = " << inner_ite_dt_s << "\n";

				if (number_of_iterations % restart_output_interval == 0 && number_of_iterations != sph_system.restart_step_)
					restart_io.writeToFile(number_of_iterations);
			}
			number_of_iterations++;

			/** Water block configuration and periodic condition. */
			periodic_condition.bounding_.parallel_exec();

			water_block.updateCellLinkedList();
			periodic_condition.update_cell_linked_list_.parallel_exec();
			water_block_complex.updateConfiguration();
			/** one need update configuration after periodic condition. */
			insert_body.updateCellLinkedList();
			insert_body_contact.updateConfiguration();
			/** write run-time observation into file */
			write_beam_tip_displacement.writeToFile(number_of_iterations);
		}

		tick_count t2 = tick_count::now();
		/** write run-time observation into file */
		compute_vorticity.parallel_exec();
		write_real_body_states.writeToFile();
		write_total_viscous_force_on_insert_body.writeToFile(number_of_iterations);
		fluid_observer_contact.updateConfiguration();
		write_fluid_velocity.writeToFile(number_of_iterations);
		tick_count t3 = tick_count::now();
		interval += t3 - t2;
	}
	tick_count t4 = tick_count::now();

	tick_count::interval_t tt;
	tt = t4 - t1 - interval;
	std::cout << "Total wall time for computation: " << tt.seconds() << " seconds." << std::endl;

	if (sph_system.generate_regression_data_)
	{
		// The lift force at the cylinder is very small and not important in this case.
		write_total_viscous_force_on_insert_body.generateDataBase({1.0e-2, 1.0e-2}, {1.0e-2, 1.0e-2});
		write_beam_tip_displacement.generateDataBase(1.0e-2);
	}
	else
	{
		write_total_viscous_force_on_insert_body.newResultTest();
		write_beam_tip_displacement.newResultTest();
	}

	return 0;
}
