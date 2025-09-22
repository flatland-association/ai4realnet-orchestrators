# NOTE: Uncomment and implement the test runners you want
# Generated with https://github.com/flatland-association/flatland-benchmarks/blob/main/definitions/ai4realnet/gen_ai4realnet_benchmarks_sql.py
# from https://inesctecpt.sharepoint.com/:x:/r/sites/AI4REALNET/Shared%20Documents/General/WP4%20-%20Validation%20and%20impact%20assessment/Validation%20campaigns/Overview%20tests%20for%20KPI%20on%20validation%20campaign%20hub.xlsx?d=w947339379458465eaaf243a750315375&csf=1&web=1&e=RnrCdf
from ai4realnet_orchestrators.orchestrator import Orchestrator
from ai4realnet_orchestrators.railway.test_runner_kpi_pf_026_railway import TestRunner_KPI_PF_026_Railway

railway_orchestrator = Orchestrator(
  test_runners={

    # KPI-PF-026: Punctuality (Railway)
    "98ceb866-5479-47e6-a735-81292de8ca65": TestRunner_KPI_PF_026_Railway(
      test_id="98ceb866-5479-47e6-a735-81292de8ca65", scenario_ids=[
        '5a60713d-01f2-4d32-9867-21904629e254', '0db72a40-43e8-477b-89b3-a7bd1224660a',
      ], benchmark_id="3b1bdca6-ed90-4938-bd63-fd657aa7dcd7"
    ),

  }
)
