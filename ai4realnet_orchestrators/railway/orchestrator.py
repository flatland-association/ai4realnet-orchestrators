# based on https:#github.com/codalab/codabench/blob/develop/compute_worker/compute_worker.py
import logging
import os
import ssl
from typing import List

from celery import Celery
from fab_clientlib import ApiClient, DefaultApi, Configuration

from ai4realnet_orchestrators.orchestrator import Orchestrator
# NOTE: import YourTestRunner implementations here:
from ai4realnet_orchestrators.railway.test_runner_kpi_af_029_railway import TestRunner_KPI_AF_029_Railway
from ai4realnet_orchestrators.railway.test_runner_kpi_nf_045_railway import TestRunner_KPI_NF_045_Railway
from ai4realnet_orchestrators.railway.test_runner_kpi_pf_026_railway import TestRunner_KPI_PF_026_Railway

logger = logging.getLogger(__name__)

app = Celery(
    broker=os.environ.get('BROKER_URL'),
    backend=os.environ.get('BACKEND_URL'),
    queue=os.environ.get("BENCHMARK_ID"),
    broker_use_ssl={
        'keyfile': os.environ.get("RABBITMQ_KEYFILE"),
        'certfile': os.environ.get("RABBITMQ_CERTFILE"),
        'ca_certs': os.environ.get("RABBITMQ_CA_CERTS"),
        'cert_reqs': ssl.CERT_REQUIRED
    }
)

# NOTE: Uncomment and implement the test runners you want
# Generated with https://github.com/flatland-association/flatland-benchmarks/blob/main/definitions/ai4realnet/gen_ai4realnet_benchmarks_sql.py
# from https://inesctecpt.sharepoint.com/:x:/r/sites/AI4REALNET/Shared%20Documents/General/WP4%20-%20Validation%20and%20impact%20assessment/Validation%20campaigns/Overview%20tests%20for%20KPI%20on%20validation%20campaign%20hub.xlsx?d=w947339379458465eaaf243a750315375&csf=1&web=1&e=RnrCdf
railway_orchestrator = Orchestrator(
    test_runners={

        #     # KPI-AS-001: Ability to anticipate (Railway)
        #     "8edbf9a1-09b4-478a-ac53-8fc903e70cc1": TestRunner_KPI_AS_001_Railway(
        #         test_id="8edbf9a1-09b4-478a-ac53-8fc903e70cc1", scenario_ids=['c7159f8d-781b-40fc-9efa-cd0e3a8b8d21'], benchmark_id="3237ba20-ccff-45b0-af23-44719e584f41"
        #     ),
        #
        #     # KPI-AS-009: Assistant disturbance (Railway)
        #     "d4bcbd11-740a-478f-8dbc-6d9b53fa6059": TestRunner_KPI_AS_009_Railway(
        #         test_id="d4bcbd11-740a-478f-8dbc-6d9b53fa6059", scenario_ids=['59d015ca-ca8a-4015-a113-35c182b301e4'], benchmark_id="3237ba20-ccff-45b0-af23-44719e584f41"
        #     ),
        #
        #     # KPI-DS-015: Decision support satisfaction (Railway)
        #     "22c590e5-01ad-4e96-9040-29a13ac9118f": TestRunner_KPI_DS_015_Railway(
        #         test_id="22c590e5-01ad-4e96-9040-29a13ac9118f", scenario_ids=['cbfe8dd3-e8df-464f-92cd-0adeab4a18b8'], benchmark_id="3237ba20-ccff-45b0-af23-44719e584f41"
        #     ),
        #
        #     # KPI-HS-022: Human motivation (Railway)
        #     "05e91ee6-744b-47ea-85a0-005e26b578e0": TestRunner_KPI_HS_022_Railway(
        #         test_id="05e91ee6-744b-47ea-85a0-005e26b578e0", scenario_ids=['ab770463-5832-4edd-9ab2-178b7ee46b74'], benchmark_id="3237ba20-ccff-45b0-af23-44719e584f41"
        #     ),
        #
        #     # KPI-HS-023: Human response time (Railway)
        #     "40398221-b288-46d9-895c-45d278f1b6a3": TestRunner_KPI_HS_023_Railway(
        #         test_id="40398221-b288-46d9-895c-45d278f1b6a3", scenario_ids=['35cf47d4-83f4-4eb7-ab9b-40c6fe7679a8'], benchmark_id="3237ba20-ccff-45b0-af23-44719e584f41"
        #     ),
        #
        #     # KPI-SS-031: Situation awareness (Railway)
        #     "307bce4c-635f-44c3-9ed2-6d4d45d1bbb2": TestRunner_KPI_SS_031_Railway(
        #         test_id="307bce4c-635f-44c3-9ed2-6d4d45d1bbb2", scenario_ids=['3131063f-5d0e-42bd-955d-5e875ceaac94'], benchmark_id="3237ba20-ccff-45b0-af23-44719e584f41"
        #     ),
        #
        #     # KPI-WS-040: Workload (Railway)
        #     "343bc5cf-0bab-4fdf-9760-266c5a738b08": TestRunner_KPI_WS_040_Railway(
        #         test_id="343bc5cf-0bab-4fdf-9760-266c5a738b08", scenario_ids=['c3ab2e6b-e8ed-420e-b6d2-4fd5dd407288'], benchmark_id="3237ba20-ccff-45b0-af23-44719e584f41"
        #     ),
        #
        #     # KPI-CS-049: Cognitive Performance & Stress (Railway)
        #     "452bb0df-9a9d-4475-ac7b-8e62659c0b13": TestRunner_KPI_CS_049_Railway(
        #         test_id="452bb0df-9a9d-4475-ac7b-8e62659c0b13", scenario_ids=['76d73b49-ff52-4b9a-96e2-cc664582c8e4'], benchmark_id="3237ba20-ccff-45b0-af23-44719e584f41"
        #     ),
        #
        #     # KPI-AS-002: Acceptance (Railway)
        #     "ba7fabe3-6f42-4cc5-875e-b8fb45aa17f3": TestRunner_KPI_AS_002_Railway(
        #         test_id="ba7fabe3-6f42-4cc5-875e-b8fb45aa17f3", scenario_ids=['0d85a097-9170-42aa-99ea-d56b833c27cf'], benchmark_id="2da3781a-25a9-4c89-8b43-9269844f3fef"
        #     ),
        #
        #     # KPI-AS-005: Agreement score (Railway)
        #     "fd01a013-c5aa-4ab2-9c01-ec608013c929": TestRunner_KPI_AS_005_Railway(
        #         test_id="fd01a013-c5aa-4ab2-9c01-ec608013c929", scenario_ids=['9db20a76-763a-4597-9586-cb217981e191'], benchmark_id="2da3781a-25a9-4c89-8b43-9269844f3fef"
        #     ),
        #
        #     # KPI-CS-013: Comprehensibility (Railway)
        #     "3c249970-1e83-4faa-bcf4-9fdb63ed904d": TestRunner_KPI_CS_013_Railway(
        #         test_id="3c249970-1e83-4faa-bcf4-9fdb63ed904d", scenario_ids=['0ee42e0f-a284-4979-86e8-4e50c9bfcef7'], benchmark_id="2da3781a-25a9-4c89-8b43-9269844f3fef"
        #     ),
        #
        #     # KPI-TS-038: Trust in AI solutions score (Railway)
        #     "57134a8f-d2bb-4a49-975c-4f6e1e07eb09": TestRunner_KPI_TS_038_Railway(
        #         test_id="57134a8f-d2bb-4a49-975c-4f6e1e07eb09", scenario_ids=['7b07ae08-9153-42cd-b1e9-6c03f3c1df31'], benchmark_id="2da3781a-25a9-4c89-8b43-9269844f3fef"
        #     ),
        #
        #     # KPI-TS-039: Trust towards the AI tool (Railway)
        #     "51780641-4f2a-4095-b230-fdddc4bf31af": TestRunner_KPI_TS_039_Railway(
        #         test_id="51780641-4f2a-4095-b230-fdddc4bf31af", scenario_ids=['0e9f4bb5-6e39-4ccb-b5ae-81b9a1f91607'], benchmark_id="2da3781a-25a9-4c89-8b43-9269844f3fef"
        #     ),
        #
        #     # KPI-HS-003: Human intervention frequency (Railway)
        #     "22b1dc11-7d3f-4219-adc9-b1eba58562a7": TestRunner_KPI_HS_003_Railway(
        #         test_id="22b1dc11-7d3f-4219-adc9-b1eba58562a7", scenario_ids=['6c383eec-31cd-4f3c-9296-cc5bb0d7f4c9'], benchmark_id="8d5c876e-22c2-49e7-bdd5-4c1840d309f0"
        #     ),
        #
        #     # KPI-SS-030: Significance of human revisions (Railway)
        #     "24eecb84-e881-459c-8116-a224b0253b70": TestRunner_KPI_SS_030_Railway(
        #         test_id="24eecb84-e881-459c-8116-a224b0253b70", scenario_ids=['eee4453e-477f-4057-ad7b-c1c2233ed108'], benchmark_id="8d5c876e-22c2-49e7-bdd5-4c1840d309f0"
        #     ),
        #
        #     # KPI-PS-089: Perceived decision novelty (Railway)
        #     "74c88d9b-f61b-4162-aefc-75a9d538b8a6": TestRunner_KPI_PS_089_Railway(
        #         test_id="74c88d9b-f61b-4162-aefc-75a9d538b8a6", scenario_ids=['b228ee9c-de35-4efd-b59a-7c4bb95ca127'], benchmark_id="8d5c876e-22c2-49e7-bdd5-4c1840d309f0"
        #     ),
        #
        #     # KPI-AS-006: AI co-learning capability (Railway)
        #     "a38618ca-6e9f-437d-a177-270f721ad1c6": TestRunner_KPI_AS_006_Railway(
        #         test_id="a38618ca-6e9f-437d-a177-270f721ad1c6", scenario_ids=['9fdd3565-45d6-4d83-9cda-478c56e94f26'], benchmark_id="65547935-f436-49fa-8d20-f320c6bd46dc"
        #     ),
        #
        #     # KPI-HS-021: Human learning (Railway)
        #     "c258c64f-1905-4d7f-93f0-d696c133978e": TestRunner_KPI_HS_021_Railway(
        #         test_id="c258c64f-1905-4d7f-93f0-d696c133978e", scenario_ids=['b4c9184c-c0d1-4a0c-bab1-9210bf8cb548'], benchmark_id="65547935-f436-49fa-8d20-f320c6bd46dc"
        #     ),
        #
        #     # KPI-DF-016: Delay reduction efficiency (Railway)
        #     "6ff3c588-357c-41a6-a45a-2bd946b158c8": TestRunner_KPI_DF_016_Railway(
        #         test_id="6ff3c588-357c-41a6-a45a-2bd946b158c8", scenario_ids=['ba7f9aac-5e96-4436-bae1-23629c4d153b'], benchmark_id="3b1bdca6-ed90-4938-bd63-fd657aa7dcd7"
        #     ),
        #
        # KPI-PF-026: Punctuality (Railway)
        "98ceb866-5479-47e6-a735-81292de8ca65": TestRunner_KPI_PF_026_Railway(
            test_id="98ceb866-5479-47e6-a735-81292de8ca65", scenario_ids=['5a60713d-01f2-4d32-9867-21904629e254'],
            benchmark_id="3b1bdca6-ed90-4938-bd63-fd657aa7dcd7"
        ),

        # KPI-AF-029: AI Response time (Railway)
        "1e226684-a836-468d-9929-b95bbf2f88dc": TestRunner_KPI_AF_029_Railway(
            test_id="1e226684-a836-468d-9929-b95bbf2f88dc", scenario_ids=['c5219c2e-c3b9-4e7a-aefc-b767a9b3005d'],
            benchmark_id="3b1bdca6-ed90-4938-bd63-fd657aa7dcd7"
        ),

        # KPI-NF-045: Network Impact Propagation (Railway)
        "e075d4a7-5cda-4d3c-83ac-69a0db1d74dd": TestRunner_KPI_NF_045_Railway(
            test_id="e075d4a7-5cda-4d3c-83ac-69a0db1d74dd", scenario_ids=['bb6302f1-0dc2-43ed-976b-4e5d3126006a'],
            benchmark_id="4b0be731-8371-4e4e-a673-b630187b0bb8"
        ),
        #
        #     # KPI-HS-018: Human control/autonomy over the process (Railway)
        #     "6a896809-dad8-4248-a2f3-d54373953fe6": TestRunner_KPI_HS_018_Railway(
        #         test_id="6a896809-dad8-4248-a2f3-d54373953fe6", scenario_ids=['dddb0b01-6bab-408e-9b12-4d7ab8e3b542'], benchmark_id="d65cd37a-4830-468c-9100-0f60ee9ff72e"
        #     ),
        #
        #     # KPI-IS-041: Impact on workload (Railway)
        #     "0d901459-427d-43ea-9c97-3815eaa52bf6": TestRunner_KPI_IS_041_Railway(
        #         test_id="0d901459-427d-43ea-9c97-3815eaa52bf6", scenario_ids=['e967a3b5-6d9b-41c2-9897-4e40a186e879'], benchmark_id="d65cd37a-4830-468c-9100-0f60ee9ff72e"
        #     ),
        #
        #     # KPI-AF-050: AI-Agent Scalability Training (Railway)
        #     "dff7e358-ff14-45e7-bc22-aac2b50500f3": TestRunner_KPI_AF_050_Railway(
        #         test_id="dff7e358-ff14-45e7-bc22-aac2b50500f3", scenario_ids=['d7cea956-6803-488c-b402-079d13b892c6'], benchmark_id="16706c82-75df-4969-932d-a7f5c941eca2"
        #     ),
        #
        #     # KPI-AF-051: AI-Agent Scalability Testing (Railway)
        #     "b2e91a79-1390-414f-bf5d-8a6fd93c6080": TestRunner_KPI_AF_051_Railway(
        #         test_id="b2e91a79-1390-414f-bf5d-8a6fd93c6080", scenario_ids=['1f2b1af1-dc36-49ae-9322-e61656951545'], benchmark_id="16706c82-75df-4969-932d-a7f5c941eca2"
        #     ),
        #
        #     # KPI-DF-052: Domain shift adaptation time (Railway)
        #     "0c3b9843-1940-45ef-943a-dc13ec1d090a": TestRunner_KPI_DF_052_Railway(
        #         test_id="0c3b9843-1940-45ef-943a-dc13ec1d090a", scenario_ids=['adb8b61b-6203-436d-9b40-8d4f95af7f43'], benchmark_id="43040944-39ac-47c9-b91d-bc8ca5693b3c"
        #     ),
        #
        #     # KPI-DF-053: Domain shift generalization gap (Railway)
        #     "fba17d1f-2eb7-4c06-8f77-5da14e7b875a": TestRunner_KPI_DF_053_Railway(
        #         test_id="fba17d1f-2eb7-4c06-8f77-5da14e7b875a", scenario_ids=['d741cc0b-7dd9-4c2e-b23c-805b622c3878'], benchmark_id="43040944-39ac-47c9-b91d-bc8ca5693b3c"
        #     ),
        #
        #     # KPI-DF-054: Domain shift out of domain detection accuracy (Railway)
        #     "69259fb3-bba8-4e62-9103-552b439ebaf4": TestRunner_KPI_DF_054_Railway(
        #         test_id="69259fb3-bba8-4e62-9103-552b439ebaf4", scenario_ids=['8d292cff-0409-4ec7-83fb-3034c3a4106c'], benchmark_id="43040944-39ac-47c9-b91d-bc8ca5693b3c"
        #     ),
        #
        #     # KPI-DF-055: Domain shift policy robustness (Railway)
        #     "dce1c0e9-f2ae-4faa-a6b3-76d037c082d0": TestRunner_KPI_DF_055_Railway(
        #         test_id="dce1c0e9-f2ae-4faa-a6b3-76d037c082d0", scenario_ids=['4b42d9b8-9f27-4247-8bef-47ad3ce3a978'], benchmark_id="43040944-39ac-47c9-b91d-bc8ca5693b3c"
        #     ),
        #
        #     # KPI-DF-056: Domain shift robustness to domain parameters (Railway)
        #     "03e1fcaf-a933-4ad1-abf2-5541574298a2": TestRunner_KPI_DF_056_Railway(
        #         test_id="03e1fcaf-a933-4ad1-abf2-5541574298a2", scenario_ids=['2238fa19-e337-49ce-a3b3-08abf85f1453'], benchmark_id="43040944-39ac-47c9-b91d-bc8ca5693b3c"
        #     ),
        #
        #     # KPI-DF-057: Domain shift success rate drop (Railway)
        #     "d795c7bd-f8be-43c2-ae6e-a931194d3fc5": TestRunner_KPI_DF_057_Railway(
        #         test_id="d795c7bd-f8be-43c2-ae6e-a931194d3fc5", scenario_ids=['5d017ca4-f062-44c9-821c-31f85928903a'], benchmark_id="43040944-39ac-47c9-b91d-bc8ca5693b3c"
        #     ),
        #
        #     # KPI-DF-090: Domain shift forgetting rate (Railway)
        #     "511f2ab0-da90-4d55-a23f-af5eda0baf7d": TestRunner_KPI_DF_090_Railway(
        #         test_id="511f2ab0-da90-4d55-a23f-af5eda0baf7d", scenario_ids=['cf7bb259-0ad4-4454-a9c8-eb8add0bec57'], benchmark_id="43040944-39ac-47c9-b91d-bc8ca5693b3c"
        #     ),
        #
        #     # KPI-RS-058: Robustness to operator input (Railway)
        #     "1cbf44c3-0c82-4f9e-9857-c7c1d96d3ab9": TestRunner_KPI_RS_058_Railway(
        #         test_id="1cbf44c3-0c82-4f9e-9857-c7c1d96d3ab9", scenario_ids=['7a1c9dac-ec75-42e1-9355-34d88eabc52f'], benchmark_id="3810191b-8cfd-4b03-86b2-f7e530aab30d"
        #     ),
        #
        #     # KPI-DF-069: Drop-off in reward (Railway)
        #     "a94c858e-4bc3-4d67-bd78-5c81506e39f7": TestRunner_KPI_DF_069_Railway(
        #         test_id="a94c858e-4bc3-4d67-bd78-5c81506e39f7", scenario_ids=['74dd5830-6e59-423f-89f4-b050319db14e'], benchmark_id="3810191b-8cfd-4b03-86b2-f7e530aab30d"
        #     ),
        #
        #     # KPI-FF-070: Frequency changed output AI agent (Railway)
        #     "5abadf6b-991c-4d37-810f-f77bb71d490d": TestRunner_KPI_FF_070_Railway(
        #         test_id="5abadf6b-991c-4d37-810f-f77bb71d490d", scenario_ids=['ffcadd8d-207a-49af-8b09-54e922642f01'], benchmark_id="3810191b-8cfd-4b03-86b2-f7e530aab30d"
        #     ),
        #
        #     # KPI-SF-071: Severity of changed output AI agent (Railway)
        #     "dce32e78-e827-4994-a0a2-06feee2528cc": TestRunner_KPI_SF_071_Railway(
        #         test_id="dce32e78-e827-4994-a0a2-06feee2528cc", scenario_ids=['588ae37c-f583-47df-9154-ca12c9ac134a'], benchmark_id="3810191b-8cfd-4b03-86b2-f7e530aab30d"
        #     ),
        #
        #     # KPI-SF-072: Steps survived with perturbations (Railway)
        #     "e5206c56-75a0-41fa-9db3-bec66359337e": TestRunner_KPI_SF_072_Railway(
        #         test_id="e5206c56-75a0-41fa-9db3-bec66359337e", scenario_ids=['8011c7bd-6082-4653-8d9d-887d23f1ec5c'], benchmark_id="3810191b-8cfd-4b03-86b2-f7e530aab30d"
        #     ),
        #
        #     # KPI-VF-073: Vulnerability to perturbation (Railway)
        #     "0ddba8a7-5ef8-45d1-b0d6-0842bc44d2cc": TestRunner_KPI_VF_073_Railway(
        #         test_id="0ddba8a7-5ef8-45d1-b0d6-0842bc44d2cc", scenario_ids=['47a11418-fad5-4d55-a637-6b90a8351500'], benchmark_id="3810191b-8cfd-4b03-86b2-f7e530aab30d"
        #     ),
        #
        #     # KPI-RF-078: Reward per action (Railway)
        #     "8ebc88f0-896c-4910-8997-a44d107e7eb7": TestRunner_KPI_RF_078_Railway(
        #         test_id="8ebc88f0-896c-4910-8997-a44d107e7eb7", scenario_ids=['dc8195e4-266d-4afb-ba60-2659f59acfa4'], benchmark_id="3810191b-8cfd-4b03-86b2-f7e530aab30d"
        #     ),
        #
        #     # KPI-AF-074: Area between reward curves (Railway)
        #     "707a1a4e-7073-432b-94fc-af4a5ee9f07d": TestRunner_KPI_AF_074_Railway(
        #         test_id="707a1a4e-7073-432b-94fc-af4a5ee9f07d", scenario_ids=['a9e3fbf7-b5d5-477d-a0c8-d23880237d2d'], benchmark_id="31ea606b-681a-437a-85b9-7c81d4ccc287"
        #     ),
        #
        #     # KPI-DF-075: Degradation time (Railway)
        #     "2c4be118-6108-43b3-b09f-a4bee842167a": TestRunner_KPI_DF_075_Railway(
        #         test_id="2c4be118-6108-43b3-b09f-a4bee842167a", scenario_ids=['dc03b9f1-bfb2-44b8-b124-f7eede10e0a7'], benchmark_id="31ea606b-681a-437a-85b9-7c81d4ccc287"
        #     ),
        #
        #     # KPI-RF-076: Restorative time (Railway)
        #     "2cac54e0-aaf3-4f22-8307-f23878c432f0": TestRunner_KPI_RF_076_Railway(
        #         test_id="2cac54e0-aaf3-4f22-8307-f23878c432f0", scenario_ids=['56160b90-a287-4dec-acc7-f40967d60fa0'], benchmark_id="31ea606b-681a-437a-85b9-7c81d4ccc287"
        #     ),
        #
        #     # KPI-SF-077: Similarity state to unperturbed situation (Railway)
        #     "d432299f-dbee-46ba-9e15-77954086440a": TestRunner_KPI_SF_077_Railway(
        #         test_id="d432299f-dbee-46ba-9e15-77954086440a", scenario_ids=['367a8b12-1b87-42f1-9400-4ecf96d6b617'], benchmark_id="31ea606b-681a-437a-85b9-7c81d4ccc287"
        #     ),
        #
        #     # KPI-RS-091: Reflection on operator trust  (Railway)
        #     "9e680ab9-c861-4dd7-a5ea-abbdc1a91088": TestRunner_KPI_RS_091_Railway(
        #         test_id="9e680ab9-c861-4dd7-a5ea-abbdc1a91088", scenario_ids=['e9e537cc-a8d0-4864-9460-9098c72b269f'], benchmark_id="df309815-8ec0-4a6f-9d0b-dc3dbfc9055a"
        #     ),
        #
        #     # KPI-RS-092: Reflection on operator agency  (Railway)
        #     "8eb804ef-d122-4c59-a6b3-53f3c9664d2b": TestRunner_KPI_RS_092_Railway(
        #         test_id="8eb804ef-d122-4c59-a6b3-53f3c9664d2b", scenario_ids=['1ce89667-c4c2-483d-8ed1-c7add0a24b4b'], benchmark_id="df309815-8ec0-4a6f-9d0b-dc3dbfc9055a"
        #     ),
        #
        #     # KPI-RS-093: Reflection on operator de-skilling  (Railway)
        #     "8f5a1908-6319-4e2a-a43b-6528c7c5e92e": TestRunner_KPI_RS_093_Railway(
        #         test_id="8f5a1908-6319-4e2a-a43b-6528c7c5e92e", scenario_ids=['e283be95-96da-4f69-8869-06ab82ee2c45'], benchmark_id="df309815-8ec0-4a6f-9d0b-dc3dbfc9055a"
        #     ),
        #
        #     # KPI-RS-094: Reflection on over-reliance  (Railway)
        #     "a6a54b1d-54bf-4b09-9887-706eed3e1a57": TestRunner_KPI_RS_094_Railway(
        #         test_id="a6a54b1d-54bf-4b09-9887-706eed3e1a57", scenario_ids=['fbe33158-fb85-492d-9015-62b95f4658e8'], benchmark_id="df309815-8ec0-4a6f-9d0b-dc3dbfc9055a"
        #     ),
        #
        #     # KPI-RS-095: Reflection on additional training  (Railway)
        #     "0d898adb-7378-4ddd-8afa-a7136f13e3fa": TestRunner_KPI_RS_095_Railway(
        #         test_id="0d898adb-7378-4ddd-8afa-a7136f13e3fa", scenario_ids=['4a6d19a8-0d27-42d8-97af-9197987e64ce'], benchmark_id="df309815-8ec0-4a6f-9d0b-dc3dbfc9055a"
        #     ),
        #
        #     # KPI-RS-096: Reflection on biases  (Railway)
        #     "be770a02-c8be-4774-97b6-bc15e91b2fd8": TestRunner_KPI_RS_096_Railway(
        #         test_id="be770a02-c8be-4774-97b6-bc15e91b2fd8", scenario_ids=['b646c5dd-20e5-4628-984c-0b6ab3156538'], benchmark_id="df309815-8ec0-4a6f-9d0b-dc3dbfc9055a"
        #     ),
        #
        #     # KPI-PS-097: Predicted long-term adoption  (Railway)
        #     "2c71e014-3994-4413-a469-e27b0d1e121d": TestRunner_KPI_PS_097_Railway(
        #         test_id="2c71e014-3994-4413-a469-e27b0d1e121d", scenario_ids=['691ab2fd-9827-45f9-8d03-0878c1005144'], benchmark_id="df309815-8ec0-4a6f-9d0b-dc3dbfc9055a"
        #     ),
        #
    }
)


# https:#docs.celeryq.dev/en/stable/userguide/tasks.html#bound-tasks: A task being bound means the first argument to the task will always be the task instance (self).
# https:#docs.celeryq.dev/en/stable/userguide/tasks.html#names: Every task must have a unique name.
@app.task(name="Railway", bind=True)
def orchestrator(self, submission_data_url: str, tests: List[str] = None, **kwargs):
    submission_id = self.request.id
    benchmark_id = orchestrator.name
    logger.info(
        f"Queue/task {benchmark_id} received submission {submission_id} with submission_data_url={submission_data_url} for tests={tests}"
    )
    # fail fast
    check_fab_healthy()
    return railway_orchestrator.run(
        submission_id=submission_id,
        submission_data_url=submission_data_url,
        tests=tests,
    )


def check_fab_healthy():
    FAB_API_URL = os.environ.get("FAB_API_URL")
    fab = DefaultApi(ApiClient(configuration=Configuration(host=FAB_API_URL)))
    print(fab.health_live_get())
