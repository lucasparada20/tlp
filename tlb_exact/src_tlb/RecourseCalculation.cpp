#include "RecourseCalculation.h"
#include <random> //std::random_device

#define STATION_LOWER_BOUND 1
#define STATION_UPPER_BOUND 2

// Add the user-defined reduction declarations for std::vectors
#pragma omp declare reduction(+:std::vector<int>: \
            std::transform(omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(), std::plus<int>())) \
            initializer(omp_priv=omp_orig)

#pragma omp declare reduction(+:std::vector<double>: \
            std::transform(omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(), std::plus<double>())) \
            initializer(omp_priv=omp_orig)

RecourseCalculation::RecourseCalculation(Scenarios * _scs, Prob * _prob) : scs(_scs), prob(_prob), user_defined_threads(0)
{
	if(prob==0 || scs==0)
	{
		printf("No scenarios in to compute recourse. Phil Collins (1989)\n"); exit(1);
	}

	#ifdef USE_OMP
		if (getenv("OMP_NUM_THREADS") == NULL) {
			printf("Warning: OMP_NUM_THREADS environment variable is not set If you are in Compute Canada you need to set this value.\n");
		}
		printf("Computation of the recourse will be done in parallel with %d threads\n", omp_get_max_threads());
	#else
		printf("OpenMP is not enabled.\n");
	#endif
	
	
	graphObjects.resize(scs->GetScenarioCount(),0);	
	double start_time = omp_get_wtime();
	total_node_count = 0; total_arc_count = 0; total_trip_count = 0;
	
	#ifdef USE_OMP
		#pragma omp parallel for reduction(+:total_node_count) reduction(+:total_arc_count) reduction(+:total_trip_count)
	#endif	
	for(int e=0;e<scs->GetScenarioCount();e++)
	{
		double start_loop_time = omp_get_wtime();
		graphObjects[e] = new ScenarioGraph(prob,scs->GetScenarioObject(e));
		
		double end_loop_time = omp_get_wtime();
		double elapsedSecondsLoop = end_loop_time - start_loop_time;
		printf("Elapsed time to build scenario:%d graph:%.1lf Nodes:%d HoldingArcs:%d TripArcs:%d TotalArcs:%d Size of arc:%lu\n",e,elapsedSecondsLoop,graphObjects[e]->GetNodeCount(),graphObjects[e]->GetNetworkArcCount(),graphObjects[e]->GetODtripCount(),graphObjects[e]->GetArcCount(),sizeof(MCF_arc));
		
		total_node_count += graphObjects[e]->GetNodeCount();
		total_arc_count += graphObjects[e]->GetNetworkArcCount() + graphObjects[e]->GetODtripCount();
		total_trip_count += graphObjects[e]->GetODtripCount();
		//char filename[100];
		//sprintf(filename,"NewGraph%d.dot",e);
		//graphObjects[e]->PrintGraph(filename);
	}	
	double end_time = omp_get_wtime();
	double elapsedSeconds = end_time - start_time;
	printf("Elapsed time to build ALL required mcf graphs:%.1lf\n",elapsedSeconds);
	printf("Total Nodes:%d Total Arcs:%d Avg Nodes:%.1lf Arcs:%.1lf AvgTrips:%.2lf ... \n",total_node_count,total_arc_count,total_node_count/(double)scs->GetScenarioCount(),total_arc_count/(double)scs->GetScenarioCount(),total_trip_count/(double)scs->GetScenarioCount());
	graph_type = 7; // SingleSourceRecourse
	printf("Recourse will be computed with graph type:%s\n",graph_type == 4 ? "Multiple Source Recourse" : "Single Source Recourse"  );
	//exit(1);
}

RecourseCalculation::~RecourseCalculation()
{
	if(graphObjects.size()>1)
		for(int e = 0; e < scs->GetScenarioCount(); e++)
			if(graphObjects[e])
			{
				//printf("Deleting scenario graph %d\n",e);
				delete graphObjects[e];
			}
				
}

double RecourseCalculation::Calculate(std::vector<int> & targets, const std::vector<int> & hm, const std::vector<int> & hp)
{
	//printf("RecourseCost Calculate():\n");
	double total_startTime = omp_get_wtime();

	if(targets.size()<1 || targets.size() < prob->GetNodeCount())
	{
		printf("Insufficient or no targets given to compute the recourse. Did you forget to give the targets? exiting ... \n"); exit(1);
	}
	/*printf("Targets in RecCost Calculate:\n");
	for(int i=0;i<targets.size();i++)
		printf("t%d:%d ",i,targets[i]);
	printf("\n");*/
	unsatisfiedRequests.clear();
	unsatisfiedRequests.resize(scs->GetScenarioCount(),0.0);
	SceNbSatisfiedTrips.clear();
	SceNbSatisfiedTrips.resize(scs->GetScenarioCount(),-1);
	
	double RecCost = 0.0;
	#ifdef USE_OMP
		#pragma omp parallel for reduction(+:RecCost)
	#endif	
	for(int e=0;e<scs->GetScenarioCount();e++)
	{	
		McfpSolvers solver;
		if(calculate_target_levels)
			solver.calculate_target_levels = true;
			
		
		if(!hm.size() && !hp.size())
			RecCost += solver.Solve(graphObjects[e], true, false, graph_type,-1,-1,targets);
		else
			RecCost += solver.Solve(graphObjects[e], false, false, OptStatCapRecourse, -1, -1, targets, hm, hp);
			
		int nb_trips = solver.GetObjectiveValue()+0.1;
		unsatisfiedRequests[e] = graphObjects[e]->GetODtripCount()-nb_trips;
		
		SceNbSatisfiedTrips[e] = solver.GetNbSatisfiedTrips();
		
		//if(show)
		//	printf("cumulativeRecCost:%.1lf RecCost:%.1lf sce:%d Primal:%.1lf Dual:%.1lf nbSatTrips:%d unsatReqs:%d nodes:%d arcs:%d trips:%d\n",RecCost/(double)scs->GetScenarioCount(),RecCost,e,graphObjects[e]->GetObjectiveValue(),graphObjects[e]->GetDualObjectiveValue(),solver.GetNbSatisfiedTrips(),unsatisfiedRequests[e],graphObjects[e]->GetNodeCount(),graphObjects[e]->GetArcCount(),graphObjects[e]->GetODtripCount()); //getchar();

	}
	double total_lastTime = omp_get_wtime();
	double elapsedSeconds = total_lastTime-total_startTime;
	//Some output ...
	//std::cout << "RecourseCost solver:" << Parameters::GetSolverName() << " Resolution total time: " << std::setprecision(3) << elapsedSeconds << " seconds" << std::endl;
		
	return RecCost / (double)scs->GetScenarioCount();
}

double RecourseCalculation::CalculateScenario(std::vector<int> &targets, int scenarioIndex, const std::vector<int> &hm, const std::vector<int> &hp)
{
    double total_startTime = omp_get_wtime();

    if (targets.size() < 1 || targets.size() < prob->GetNodeCount()) {
        printf("Insufficient or no targets given to compute the recourse. Did you forget to give the targets? exiting ... \n");
        exit(1);
    }

    double RecCost = 0.0;
    
    // Ensure scenarioIndex is valid
    if (scenarioIndex < 0 || scenarioIndex >= scs->GetScenarioCount()) {
        printf("Invalid scenario index: %d. Exiting...\n", scenarioIndex);
        exit(1);
    }

    McfpSolvers solver;
    if (!hm.size() && !hp.size())
        RecCost += solver.Solve(graphObjects[scenarioIndex], false, false, graph_type, -1, -1, targets);
    else
        RecCost += solver.Solve(graphObjects[scenarioIndex], false, false, OptStatCapRecourse, -1, -1, targets, hm, hp);

    // Calculate the number of trips and unsatisfied requests
    int nb_trips = solver.GetObjectiveValue() + 0.1;
    int unsatisfiedRequests = graphObjects[scenarioIndex]->GetODtripCount() - nb_trips;
    int SceNbSatisfiedTrips = solver.GetNbSatisfiedTrips();

    double total_lastTime = omp_get_wtime();
    double elapsedSeconds = total_lastTime - total_startTime;

    // std::cout << "RecourseCost solver: " << Parameters::GetSolverName() << " Resolution total time: " << std::setprecision(3) << elapsedSeconds << " seconds" << std::endl;

    return RecCost;  
}

double RecourseCalculation::Calculate(std::vector<int> & targets, std::vector<double> & duals, std::vector<double> & capacity_duals, const std::vector<int> & hm, const std::vector<int> & hp)
{
	double RecCost = 0.0;
	unsatisfiedRequests.clear();
	unsatisfiedRequests.resize(scs->GetScenarioCount(),0.0);
	capacity_duals.resize(prob->GetNodeCount(),0.0);
	/*printf("Nodes in Calculate():%d duals:%d Qtot:%d\n",(int)targets.size(),(int)duals.size(),prob->GetQtot());
	for(int i=0;i<targets.size();i++)
		printf("t%d:%d ",i,targets[i]);
	printf("\n");*/
	//getchar();
	//printf("In RecCalculate sizeOf hm:%d hp:%d\n",(int)hm.size(),(int)hp.size());

	
	#ifdef USE_OMP
		#pragma omp parallel for reduction(+:RecCost) reduction(+:duals) reduction(+:capacity_duals)
	#endif
	for(int e=0;e<scs->GetScenarioCount();e++)
	{		
		McfpSolvers solver;
		solver.y0.clear(); solver.y0.resize(prob->GetNodeCount(),0);
		for(int i=0;i<prob->GetNodeCount();i++)
			solver.y0[i] = y0[i];

		if (hm.size() || hp.size()) {
			graph_type = OptStatCapRecourse;
			RecCost += solver.Solve(graphObjects[e], false, true, graph_type, -1, -1, targets, hm, hp);
		} else {
			//By default: graph_type == SingleSourceRecourse
			//RecCost += solver.Solve(graphObjects[e], false, true, graph_type, -1, -1, targets);
			RecCost += solver.Solve(graphObjects[e], false, true, xZero, -1, -1, targets);
		}	
		
		int nb_trips = graphObjects[e]->GetObjectiveValue()+0.1;
		
		//Depot outgoing arcs 
		for(int i=0;i<prob->GetNodeCount();i++)
		{
			if(graph_type == Recourse || graph_type == OptStatCapRecourse)
				duals[i] += graphObjects[e]->pi[i+2] - graphObjects[e]->pi[0];
			else if(graph_type == SingleSourceRecourse || graph_type == xZero)
			{
				MCF_arc * a = graphObjects[e]->GetArc(i);
				duals[i] += a->dual;
			}
				
		}
		
		for(int i=prob->GetNodeCount(); i<graphObjects[e]->GetArcCount(); i++)
		{
			MCF_arc * a = graphObjects[e]->GetArc(i);
			duals[prob->GetNodeCount()] += (a->cap * a->dual);
		}
		
		double pi_diff = graphObjects[e]->pi[0] - graphObjects[e]->pi[1];
		
		double constant = prob->GetQtot() * pi_diff;

		duals[prob->GetNodeCount()] += constant;
		
		if(Parameters::GetModel()==5 || Parameters::GetModel()==6 || Parameters::GetModel()==7)
		{
			//printf("sce:%d NetworkArcs:%d\n",e,graphObjects[e]->GetNetworkArcCount()-1);
			for(int i=prob->GetNodeCount(); i<graphObjects[e]->GetNetworkArcCount()-1; i++) 
			{
				MCF_arc * a = graphObjects[e]->GetArc(i);
				int node_no = a->cust_no;
				if( node_no > prob->GetNodeCount() || a->type != 2 )
				{
					a->Show(); std::cerr << "Wrong arc in McfpSolvers. node_no:" << node_no << " Exiting." << std::endl; exit(1);
				}
				capacity_duals[ node_no ] += a->dual; 
				
				//printf("cust_no:%d\n",node_no); a->Show();
			}
			//getchar();
		}
		
		unsatisfiedRequests[e] = graphObjects[e]->GetODtripCount() - nb_trips;
			
		//printf("RecCost sce:%d Obj:%.1lf Dual:%.1lf unsatReqs:%d nodes:%d arcs:%d trips:%d\n",e,graphObjects[e]->GetObjectiveValue(),graphObjects[e]->GetDualObjectiveValue(),unsatisfiedRequests[e],graphObjects[e]->GetNodeCount(),graphObjects[e]->GetArcCount(),graphObjects[e]->GetODtripCount());	
		//getchar();
	}
	
	for(int i=0;i<duals.size();i++)
		duals[i] /= (double)scs->GetScenarioCount();
	
	for(int i=0;i<capacity_duals.size();i++)
		capacity_duals[i] /= (double)scs->GetScenarioCount();
	
	RecCost /= (double)scs->GetScenarioCount();
	
	//printf("forCut: RecCost:%.4lf\n",RecCost);
	//for(int i=0;i<duals.size();i++)
	//	printf("d%d:%.2lf ",i,duals[i]);
	//printf("\n"); //getchar();
	//printf("forCut: DualCapacities:\n",RecCost);
	//for(int i=0;i<capacity_duals.size();i++)
	//	printf("d%d:%.2lf ",i,capacity_duals[i]);
	//printf("\n"); //getchar();	
	
	return RecCost;	
}

double RecourseCalculation::CalculateParetoOpt(std::vector<int> & targets, std::vector<double> & duals)
{
	std::vector<int> yBar = targets;
	//std::vector<int> y0(targets.size(),0);
	//need to fill y0 with something ...
	//printf("Capacity->");
	//for(int i=0;i<prob->GetNodeCount();i++)
	//	printf(" %d:%d ",i,prob->GetNode(i)->stationcapacity);
	//printf("\n");	
	//printf("y0->");
	//for(int i=0;i<prob->GetNodeCount();i++)
	//{
		//int cap = prob->GetNode(i)->stationcapacity;
		//y0[i] =  std::min( (int)std::ceil(cap/2.0), cap );
		//y0[i] = WStargets[i];
		//if(y0[i]>0) printf(" %d:%d ",i,y0[i]);
	//}
	//printf("\n");
	//printf("yBar->");
	//for(int i=0;i<prob->GetNodeCount();i++)
	//	printf(" %d:%d ",i,yBar[i]);
	//printf("\n");	
	
	double RecCost = 0.0;
	#ifdef USE_OMP
		#pragma omp parallel for reduction(+:RecCost) reduction(+:duals) 
	#endif
	for(int e=0;e<scs->GetScenarioCount();e++)
	{
		IloEnv env;
		ParetoOpt solver;
		
		double RyBar = 0.0; 
		solver.Init(env,graphObjects[e],prob,y0,yBar);
		bool useMcfp = true;
		if(useMcfp)
		{
			solver.RyBar = -CalculateScenario(targets,e);
			solver.SolveMCF(env);
		}
		else
			solver.Solve(env);
		RecCost += (-1*solver.sol_value)/(double)prob->GetQtot();
		//printf("Sce:%d sol_value:%.2lf RecCost:%.2lf\n",graphObjects[e]->scenario_no,solver.sol_value, RecCost);
		
		//Depot outgoing arcs 
		for(int i=0;i<prob->GetNodeCount();i++)
		{
			MCF_arc * a = graphObjects[e]->GetArc(i);
			duals[i] += a->dual;	
		}
		for(int i=prob->GetNodeCount(); i<graphObjects[e]->GetArcCount(); i++)
		{
			MCF_arc * a = graphObjects[e]->GetArc(i);
			duals[prob->GetNodeCount()] += (a->cap * a->dual);
		}
		
		double pi_diff = graphObjects[e]->pi[1] - graphObjects[e]->pi[0];
		double constant = prob->GetQtot() * pi_diff;

		duals[prob->GetNodeCount()] += constant;
		env.end();
	}
	for(int i=0;i<duals.size();i++)
		duals[i] /= (double)scs->GetScenarioCount();
	
	//printf("Pareto Opt forCut: RecCost:%.2lf ",RecCost);
	//for(int i=0;i<duals.size();i++)
	//	printf("d%d:%.1lf ",i,duals[i]);
	//printf("\n"); 
	//getchar();
	
	return RecCost/(double)scs->GetScenarioCount();		
}

double RecourseCalculation::CalculateWS(Sol * sol)
{
	std::vector<int> nb_succesful_trips(scs->GetScenarioCount(),0.0); 
	std::vector<int> nb_unsuccesful_trips(scs->GetScenarioCount(),0.0);
	
	clock_t total_startTime = clock();
	
	int nb_scenarios_with_all_succesful=0;
	double WaitAndSee = 0.0;
	
	#ifdef USE_OMP
		#pragma omp parallel for reduction (+:WaitAndSee)
	#endif
	for(int e=0;e<scs->GetScenarioCount();e++)
	{
		McfpSolvers solver;
		
		if(Parameters::GetModel() != 5 && Parameters::GetModel()!=6 && Parameters::GetModel()!=7)
			WaitAndSee += solver.Solve(graphObjects[e], true, false, WS);
		else 
			WaitAndSee += solver.Solve(graphObjects[e], true, false, WSnoCap);
		
		solver.StoreInSol(sol,e);
		
		int nb_trips = solver.GetObjectiveValue()+0.1;
		
		nb_succesful_trips[e]=nb_trips;
		nb_unsuccesful_trips[e] = graphObjects[e]->GetODtripCount() - nb_succesful_trips[e];
		
		bool found_unsuccesful = false;
		if(nb_succesful_trips[e] == (int)graphObjects[e]->GetODtripCount())
		{
			nb_scenarios_with_all_succesful++;
		}
		
	}
	sol->SetUnsatisfiedRequestsFromWs(nb_unsuccesful_trips);	
	printf("Nb scenarios with all succesful trips in Ub calculation:%d\n",nb_scenarios_with_all_succesful);
	
	WaitAndSee /= (double)scs->GetScenarioCount();
	
	WStargets.resize(prob->GetNodeCount(),0);
	for(int i=0;i<prob->GetNodeCount();i++)
	{
		int target=0;
		for(int e=0;e<scs->GetScenarioCount();e++)
			target += sol->GetTarget(e,i);
		target = std::ceil( target / (double)scs->GetScenarioCount() );	
		WStargets[i] = target <= prob->GetNode(i)->stationcapacity ? target : prob->GetNode(i)->stationcapacity;
	}	
	
	return WaitAndSee;
}

void RecourseCalculation::CalculateGlobalTargetLbUb()
{	
	LbSceNbSatisfiedTrips.clear(); LbSceNbSatisfiedTrips.resize(scs->GetScenarioCount(),-1);
	UbSceNbSatisfiedTrips.clear(); UbSceNbSatisfiedTrips.resize(scs->GetScenarioCount(),-1);
	
	int nb_lbs_set = 0; int nb_ubs_set = 0;
	#pragma omp parallel for
	for(int e=0;e<scs->GetScenarioCount();e++)
	{
		
		int lb_sum_of_arcs = 0; int ub_sum_of_arcs = 0;
		int LbSatisfiedTrips = 0; int UbSatisfiedTrips = 0;
		
		McfpSolvers solver;
		//Lower Bound
		solver.Solve(graphObjects[e], true, false, GlobalTargetLbUb, 1);	
		for(int i=0;i<prob->GetNodeCount();i++)
		{
			int flow = solver.GetFlow(i);
			lb_sum_of_arcs += flow;
			//printf("i:%d e:%d LBbikes:%d\n",i,e,flow);
			//if(graphObjects[e]->GetArc(i)->value<0.1)
			if(flow==0)
			{
				scs->SetTargetLb(e, i,0);
				nb_lbs_set++;
			}
				
		}
		scs->SetBikeLb(e, (int)(lb_sum_of_arcs+0.1));
		LbSceNbSatisfiedTrips[e] = solver.GetNbSatisfiedTrips();

		//Upper Bound
		solver.Solve(graphObjects[e], true, false, GlobalTargetLbUb, 2);
		for(int i=0;i<prob->GetNodeCount();i++)
		{
			int flow = solver.GetFlow(i);
			ub_sum_of_arcs += flow;
			//printf("i:%d e:%d UBbikes:%d\n",i,e,flow);
			if ((flow == prob->GetNode(i)->stationcapacity && (Parameters::GetModel() == 2 || Parameters::GetModel() == 3)) ||
				((flow == 9999) && (Parameters::GetModel() == 6 || Parameters::GetModel() == 7))) 
			{
				scs->SetTargetUb(e, i, flow);
				nb_ubs_set++;
			}

		}	
		scs->SetBikeUb(e, (int)(ub_sum_of_arcs+0.1));
		UbSceNbSatisfiedTrips[e] =  solver.GetNbSatisfiedTrips();
		
		if(LbSceNbSatisfiedTrips[e]!=UbSceNbSatisfiedTrips[e])
		{
			printf("Wrong NbSatisfiedTrips in Lb, Ub graphs, exiting ...\n"); exit(1);
		}
	}
	printf("GlobalLbUb:\n");
	for(int e=0;e<scs->GetScenarioCount();e++)
		printf("sce:%d bikes[lb,ub]:[%d,%d] satisfiedTrips[lb,ub]:[%d,%d]\n",e,scs->GetBikeLb(e),scs->GetBikeUb(e),LbSceNbSatisfiedTrips[e],UbSceNbSatisfiedTrips[e]);
	
	printf("GlobalLbUb found individual %d/%d Lbs and %d/%d Ubs\n",nb_lbs_set,prob->GetNodeCount()*scs->GetScenarioCount(),nb_ubs_set,prob->GetNodeCount()*scs->GetScenarioCount());	
}

void RecourseCalculation::CalculateTargetBounds()
{
	lb_i_ub_i_time = 0;
	//int nb_threads = user_defined_threads > 0 ? user_defined_threads : omp_get_max_threads();
	int nb_threads = omp_get_max_threads();

	std::vector< std::vector<ScenarioGraph*> > mcfps(nb_threads);
	
	//This should be an emplace_back() ... The scenario graph is gigantic, so avoid the copy!
	for(int e=0;e<scs->GetScenarioCount();e++)
		mcfps[0].push_back( graphObjects[e] );

	// Populate mcfps vector with existing mcfObjects for other threads
	for(int i = 1; i < nb_threads; i++)
		mcfps[i] = mcfps[0];  // Copy the vector from the first thread

	std::vector<ToDo> todos;
	for(int e=0;e<scs->GetScenarioCount();e++)
		for (int i = 0; i < prob->GetNodeCount();i++)
		{
			ToDo td;
			td.scenario = e;
			td.station = i;
			td.bound = 1;
			if(scs->GetTargetLb(e,i)!=0)
				todos.push_back(td);
			td.bound = 2;
			if ((Parameters::GetModel() == 2 && scs->GetTargetUb(e, i) != prob->GetNode(i)->stationcapacity) || 
				(Parameters::GetModel() == 6 && scs->GetTargetUb(e, i) != 9999)) 
					todos.push_back(td);
		}

	//shuffle
	std::random_device rd;
	std::mt19937 g(rd());
	std::shuffle(todos.begin(), todos.end(), g);
	
	printf("In TargetBounds ToDos:%d threads:%d\n",(int)todos.size(),nb_threads);
	size_t completed_iterations = 0;	

	double total_start_time = omp_get_wtime(); // Start time for the entire loop
	#pragma omp parallel for shared(completed_iterations)
	for(size_t k=0;k<todos.size();k++)
	{
		int td = omp_get_thread_num();
		ScenarioGraph* mcf = mcfps[td][ todos[k].scenario ];
		
		McfpSolvers solver;
		solver.Solve(mcf, false, false, TargetBounds, todos[k].bound, todos[k].station);
		
		todos[k].value = solver.GetFlow(todos[k].station);
		todos[k].nbSatisfiedTrips = solver.GetNbSatisfiedTrips();
		
		//#pragma omp critical
		//{
			//printf("i:%d e:%d bound:%d flow:%d NbSatisfiedTrips:%d\n",todos[k].station,todos[k].scenario,todos[k].bound,solver.GetFlow(todos[k].station),todos[k].nbSatisfiedTrips); getchar();
		//}
		
        // Print elapsed time every 10% of the loop
		#pragma omp atomic 
		completed_iterations++;
		if (!todos.empty() && (completed_iterations + 1) % (todos.size() / 100 + 1) == 0) {
			double elapsed_time = omp_get_wtime() - total_start_time;
			double percentage_completed = (double)(completed_iterations + 1) * 100.0 / todos.size();
			printf("Elapsed time at %.1lf%% completed of the loop: %.2f seconds\n",
				   percentage_completed, elapsed_time);
		}

	}

	double total_end_time = omp_get_wtime(); // End time for the entire loop
	double total_linear_elapsed_time = total_end_time - total_start_time;

	printf("Total elapsed time for the entire parallel loop: %.2f seconds\n", total_linear_elapsed_time);
	lb_i_ub_i_time = total_linear_elapsed_time;
	
	
	std::vector<std::vector<int>> LbM; std::vector<std::vector<int>> UbM;
	LbM.resize(scs->GetScenarioCount(),std::vector<int>(prob->GetNodeCount(),-1));
	UbM.resize(scs->GetScenarioCount(),std::vector<int>(prob->GetNodeCount(),-1));
	
	for(size_t k=0;k<todos.size();k++)
	{
		//todos[k].Show();
		if(todos[k].bound == 1)
		{
			scs->SetTargetLb(todos[k].scenario,todos[k].station,todos[k].value);
			LbM[todos[k].scenario][todos[k].station] = todos[k].nbSatisfiedTrips;
		}
		if(todos[k].bound == 2)
		{
			scs->SetTargetUb(todos[k].scenario,todos[k].station,todos[k].value);
			UbM[todos[k].scenario][todos[k].station] = todos[k].nbSatisfiedTrips;
		}
					
	}

	//Uncomment to show output
	printf("%s\n",Parameters::GetSolverName());
	for (int i = 0; i < prob->GetNodeCount(); i++)
	{
		for(int e = 0; e < scs->GetScenarioCount(); e++)
		{
			printf("i:%d cap:%d e:%d Lb:%d Ub:%d LbSatTrips:%d UbSatTrips:%d\n",i,prob->GetNode(i)->stationcapacity,e,scs->GetTargetLb(e,i),scs->GetTargetUb(e,i),LbM[e][i],UbM[e][i]);
			if ((Parameters::GetModel() == 2 && scs->GetTargetUb(e, i) > prob->GetNode(i)->stationcapacity) || 
				(Parameters::GetModel() == 6 && scs->GetTargetUb(e, i) > prob->GetNode(i)->stationcapacity) + (int)std::ceil( Parameters::GetBudget() * prob->GetCapTot() ) || 
				scs->GetTargetLb(e, i) < 0) // Remember to change to : prob->GetNode(i)->stationcapacity) + (int)std::ceil( Parameters::GetBudget() * pr->GetCapTot() )
			{
				printf("Wrong target bound, exiting ...\n"); 
				exit(1);
			}
			if(LbM[e][i]!=UbM[e][i] && LbM[e][i] != -1 && UbM[e][i] != -1)
			{
				printf("Wrong NbSatisfiedTrips in Lb, Ub graphs, exiting ...\n"); exit(1);
			}
			//REMEBER TO RMV THIS!
			//if(scs->GetTargetLb(e,i)>scs->GetTargetUb(e,i))
			//{
				//printf("Found Lb > Ub. Exiting ...\n"); exit(1);
			//}
		}
			
		printf("\n");
	}	
}

void RecourseCalculation::CalculateMinMaxBounds()
{
	printf("In CalculateMinMaxBounds:\n");
	std::vector< std::vector<int> > h_si_forLb(scs->GetScenarioCount(),std::vector<int>(prob->GetNodeCount()));    //contains a heuristic solution for station i and scenario e
	std::vector< std::vector<int> > h_si_forUb(scs->GetScenarioCount(),std::vector<int>(prob->GetNodeCount()));    //contains a heuristic solution for station i and scenario e
	
	#pragma omp parallel for
	for(int e=0;e<scs->GetScenarioCount();e++)
	{
		McfpSolvers solver;
		//finds the minimal number of required bikes for scenario e		
		solver.Solve(graphObjects[e], true, false, GlobalTargetLbUb, 1);	
		for(int i=0;i<prob->GetNodeCount();i++)
		{
			h_si_forLb[e][i] = solver.GetFlow(i);
			//printf("h_si_forLb[%d][%d]:%d\n",e,i,h_si_forLb[e][i]);
		}
		//finds the maximal number of required bikes for scenario e	
		solver.Solve(graphObjects[e], true, false, GlobalTargetLbUb, 2);
		for(int i=0;i<prob->GetNodeCount();i++)
		{
			h_si_forUb[e][i] = solver.GetFlow(i);
			//printf("h_si_forUb[%d][%d]:%d\n",e,i,h_si_forUb[e][i]);
		}		
	}

	//clock_t start_time = clock();
	
	std::vector<int> max_or_min(prob->GetNodeCount(),-1);
	Calculate(STATION_LOWER_BOUND,h_si_forLb,max_or_min); //Computes the minimal lower bounds
	for(int i=0;i<prob->GetNodeCount();i++)
		prob->SetStationLb(i,max_or_min[i]);
	
	Calculate(STATION_UPPER_BOUND,h_si_forUb,max_or_min); //Computes the maximal upper bounds
	for(int i=0;i<prob->GetNodeCount();i++)
		prob->SetStationUb(i,max_or_min[i]);

	//clock_t end_time = clock();
	//double elapsedSeconds = ((double) end_time - start_time) / CLOCKS_PER_SEC;
	//printf("StationBounds time:%.1lf\n",elapsedSeconds);

	for(int i=0;i<prob->GetNodeCount();i++)
		printf("i:%d minLb:%d maxUb:%d cap:%d\n",i,prob->GetStationLb(i),prob->GetStationUb(i),prob->GetNode(i)->stationcapacity);
		//printf("i:%d maxLb:%d minUb:%d cap:%d InModelLb:%d InModelUb:%d\n",i,prob->GetStationLb(i),prob->GetStationUb(i),prob->GetNode(i)->stationcapacity,(int)std::floor(0.5*prob->GetStationLb(i)),(int)std::ceil(1.5*prob->GetStationUb(i)));

	//Check if lb_si < max_i, for all scenarios s
	//Check if ub_si > min_i, for all scenarios s
	//Halt if something went wrong ...
	//Uncomment to check ....
	/*for(int i = 0; i < prob->GetNodeCount(); i++) {
		int minUb = 9999999;
		int maxLb = -1;
		
		for(int e = 0; e < scs->GetScenarioCount(); e++) {
			minUb = std::min(minUb, scs->GetTargetUb(e, i));
			maxLb = std::max(maxLb, scs->GetTargetLb(e, i));
		}
		//sol->GetStationLb(i) is maximal Lb
		//sol->GetStationUb(i) is minimal Ub
		
		if(maxLb != prob->GetStationLb(i) || minUb != prob->GetStationUb(i))
		{
			printf("Wrong min_i or max_i. Exiting ... \n");
			printf("i:%d max_Lb:%d min_Ub:%d From the other method: max_Lb:%d min_Ub:%d\n", i, prob->GetStationLb(i), prob->GetStationUb(i), maxLb, minUb);			
			exit(1);
		}
	}
	printf("Pseudocode found the same bounds and the Parallel method! Great!\n");
	getchar();*/		
}

void RecourseCalculation::Calculate(int bound_type, std::vector<std::vector<int>> & h_si, std::vector<int> & max_or_min)
{

    printf("Computing station maximal %s bounds with threads:%d ... \n",( bound_type == STATION_LOWER_BOUND ) ? "LOWER" : "UPPER",omp_get_max_threads());
    std::vector<std::vector<bool>> opt_si(scs->GetScenarioCount(),std::vector<bool>(prob->GetNodeCount(),false));

	double total_startTime = omp_get_wtime();
	int completed_iterations = 0;
    #pragma omp parallel for
	for (int i = 0; i < prob->GetNodeCount(); i++)
    {
        #pragma omp atomic
		completed_iterations++;
		if ((completed_iterations + 1) % (prob->GetNodeCount() / 100 + 1) == 0) {
			double elapsed_time = omp_get_wtime() - total_startTime;
			double percentage_completed = (double)(completed_iterations + 1) * 100.0 / prob->GetNodeCount();
			printf("Bound:%s Thread:%d Elapsed time at %.1lf%% completed of the loop: %.2f seconds\n",( bound_type == 1 ) ? "Lb" : "Ub", omp_get_thread_num(),percentage_completed, elapsed_time);
		}

        while (true)
        {
            int max_e = -1;
            int min_e = -1;
            int opt_e = false;
            int min_lb = Parameters::GetModel() == 7 ? prob->GetNode(i)->stationcapacity + (int)std::ceil( Parameters::GetBudget() * prob->GetCapTot() ) : prob->GetNode(i)->stationcapacity + 1; //M7 has unlimited capacity of h_i + B
            int max_ub = -1;

            for (int e = 0; e < scs->GetScenarioCount(); e++)
            {
		if (bound_type == STATION_LOWER_BOUND)
                {
                    if (min_lb > h_si[e][i] || (min_lb == h_si[e][i] && opt_si[e][i]))
                    {
                        min_e = e;
                        opt_e = opt_si[e][i];
                        min_lb = h_si[e][i];
                    }
                }
                else if (bound_type == STATION_UPPER_BOUND)
                {
                    if (max_ub < h_si[e][i] || (max_ub == h_si[e][i] && opt_si[e][i]))
                    {
                        max_e = e;
                        opt_e = opt_si[e][i];
                        max_ub = h_si[e][i];
                    }
                }
            }
			
			/*if(!opt_e)
				printf("\nBnd:%s Stat:%d Cap:%d value:%d NotOpt, going to solve the mcfp of scenario:%d if value>0...\n",(bound_type == STATION_LOWER_BOUND)?"Lb":"Ub",i,prob->GetNode(i)->stationcapacity, (bound_type == STATION_LOWER_BOUND)? min_lb : max_ub,(bound_type == STATION_LOWER_BOUND) ? min_e : max_e);
			else printf("\nBnd:%s Stat:%d Cap:%d value:%d Opt! Skipping the mcfp due to scenario:%d\n",(bound_type == STATION_LOWER_BOUND)?"Lb":"Ub",i,prob->GetNode(i)->stationcapacity,(bound_type == STATION_LOWER_BOUND)? min_lb : max_ub,(bound_type == 1)?min_e:max_e);*/

            if (opt_e || (bound_type == STATION_LOWER_BOUND && min_lb == 0) ||
                          (bound_type == STATION_UPPER_BOUND && max_ub == prob->GetNode(i)->stationcapacity && Parameters::GetModel() == 3|| bound_type == STATION_UPPER_BOUND && max_ub == prob->GetNode(i)->stationcapacity) + (int)std::ceil( Parameters::GetBudget() * prob->GetCapTot() ) && Parameters::GetModel() == 7)
            {
                max_or_min[i] = (bound_type == STATION_LOWER_BOUND) ? min_lb : max_ub;
                break;
            }
            else
            {
                McfpSolvers solver;
				int e = (bound_type == STATION_LOWER_BOUND) ? min_e : max_e;
				if(bound_type==STATION_LOWER_BOUND)
					solver.Solve(graphObjects[e], false, false, TargetBounds, STATION_LOWER_BOUND,i);
				else
					solver.Solve(graphObjects[e], false, false, TargetBounds, STATION_UPPER_BOUND,i);
				int flow = solver.GetFlow(i);
				//printf("%d flow:%d\n",i,flow);
                h_si[e][i] = flow;
                opt_si[e][i] = true;
            }
        }
		//printf("%d %s bound:%d\n",i,(bound_type==Lb)? "Lb":"Ub", max_or_min[i]);
    }
}


struct Event
{
	int trip_idx; int16_t type; int32_t time;
	
	Event(int _trip_idx, int16_t _type, int32_t _time) : 
		trip_idx(_trip_idx),
		type(_type),
		time(_time) {}
	void Show(){ printf("Ev trip:%d type:%d time:%d\n",trip_idx,type,time); }
};

std::vector<double> RecourseCalculation::CalculateLb(std::vector<int> & targets)
{	
	std::vector<int> accepted_pickups(scs->GetScenarioCount(),0);
	std::vector<int> accepted_drops(scs->GetScenarioCount(),0);
	std::vector<int> rejected_pickups(scs->GetScenarioCount(),0);
	std::vector<int> rejected_drops(scs->GetScenarioCount(),0);
	
	printf("CalculateLb:\n");
	for(size_t i=0;i<targets.size();i++)
		printf("t%zu:%d ",i,targets[i]);
	printf("\n");
	
	for(int e=0;e<scs->GetScenarioCount();e++)
	{		
		Scenario * sce = scs->GetScenarioObject(e);
		
		std::vector<Event> events;
		events.reserve( 2*sce->GetODTripCount() );
		
		for(int i=0;i<sce->GetODTripCount();i++)
		{
			events.emplace_back( i, 1, sce->GetODTrip(i)->start_t );
			events.emplace_back( i, 2, sce->GetODTrip(i)->end_t );
		}
		
		std::sort(events.begin(), events.end(), [](const Event & ev1, const Event & ev2)
		{
			return ev1.time < ev2.time;
		});
		
		std::vector<int> bikes = targets;
		std::vector<int> reserved_spots((int)targets.size(),0);
		std::vector<bool> accepted((int)sce->GetODTripCount(),false);
		
		for(size_t i=0;i<events.size();i++)
		{
			if(events[i].trip_idx > sce->GetODTripCount())
			{
				printf("Trips:%d wrongTrip:\n",sce->GetODTripCount());
				events[i].Show(); exit(1);
			}
			
			//events[i].Show();
			//sce->GetODTrip( events[i].trip_idx )->Show();
			
			int pickup_stat = sce->GetODTrip( events[i].trip_idx )->start_no - 1;
			int drop_stat = sce->GetODTrip( events[i].trip_idx )->end_no - 1;
			
			if(pickup_stat > (int)bikes.size() || drop_stat > (int)bikes.size())
			{
				printf("p:%d d:%d bikesSize:%d\n",
						pickup_stat, drop_stat, (int)bikes.size());
				exit(1);
			}
		
			
			bool accepted_p = false; bool accepted_d = false; // To dbg
			//Accepting pickups
			if(events[i].type == 1 && 
				bikes[ pickup_stat ] > 0 &&
				bikes[ drop_stat ] + reserved_spots[ drop_stat ] + 1 <= prob->GetNode( drop_stat )->stationcapacity)
				{
					accepted_pickups[e]++;
					bikes[ pickup_stat ]--;
					reserved_spots[ drop_stat ]++;
					accepted_p = true;
					accepted[ events[i].trip_idx ] = true;
				}
				else 
				{
					if (bikes[ pickup_stat ] == 0) 
						rejected_pickups[e]++;
					if (bikes[ drop_stat ] + reserved_spots[ drop_stat ] + 1 > prob->GetNode( drop_stat )->stationcapacity)
						rejected_drops[e]++;
				}
			//Accepting drops
			if(events[i].type == 2 &&
				accepted[ events[i].trip_idx ]
				//reserved_spots[ drop_stat ] > 0 &&
				//bikes[ drop_stat ] + reserved_spots[ drop_stat ] + 1 <= prob->GetNode( drop_stat )->stationcapacity
				)
				{
					accepted_drops[e]++;
					bikes[ drop_stat ]++;
					reserved_spots[ drop_stat ]--;
					accepted_d = true;
				}			
		}
		printf("e:%d events:%zu trips:%d Accepted(p,d):(%d,%d)\n",
				e,events.size(),sce->GetODTripCount(),accepted_pickups[e],accepted_drops[e]);

	}
	
	int total_accepted_pickups = std::accumulate(accepted_pickups.begin(), accepted_pickups.end(), 0);
	int total_accepted_drops = std::accumulate(accepted_drops.begin(), accepted_drops.end(), 0);
	int total_rejected_pickups = std::accumulate(rejected_pickups.begin(), rejected_pickups.end(), 0);
	int total_rejected_drops = std::accumulate(rejected_drops.begin(), rejected_drops.end(), 0);

	
	double lb = (total_accepted_pickups + total_accepted_drops) / (2.0*(double)scs->GetScenarioCount()); // The average accepted requests
	double avg_rejected_pickups = total_rejected_pickups / (double)scs->GetScenarioCount();
	double avg_rejected_drops = total_rejected_drops / (double)scs->GetScenarioCount();
	
	printf("Avg accepted pickups:%.1lf drops:%.1lf lb:%.1lf rej_pick:%.1lf rej_drop:%.1lf\n",
			total_accepted_pickups/(double)scs->GetScenarioCount(), 
			total_accepted_drops/(double)scs->GetScenarioCount(),
			lb,
			avg_rejected_pickups,
			avg_rejected_drops);

	std::vector<double> lbs(3,-1.0);
	lbs[0] = lb;
	lbs[1] = avg_rejected_pickups;
	lbs[2] = avg_rejected_drops;
	
	return lbs;
}
