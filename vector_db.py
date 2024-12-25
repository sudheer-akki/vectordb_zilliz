from pymilvus import MilvusClient, MilvusException, utility, DataType
from typing import List
import logging
logger = logging.getLogger('dev')
logger.setLevel(logging.INFO)

class VectorDatabase:
    def __init__(self, 
                Zilliz_CLUSTER_USER,
                Zilliz_CLUSTER_PWD,
                TOKEN,
                URI,
                db_name: str ="rag_demo",
                collection_name: str="rag_collection",
                vector_field_dim: int= 384,
                metric_type: str = "COSINE"):
        self.db_name = db_name
        self.collection_name = collection_name
        self.vector_field_dim = vector_field_dim
        self.Zilliz_CLUSTER_USER = Zilliz_CLUSTER_USER
        self.Zilliz_CLUSTER_PWD = Zilliz_CLUSTER_PWD
        self.TOKEN = TOKEN
        self.URI = URI
        self.metric_type = metric_type
        self._initial_connection_setup()

    def _initial_connection_setup(self):
        self._connect_client()
        logger.info(f"Connected to {self.URI}")
        self._create_collection()

    def _connect_client(self):
        try:
            # connecting to client
            logger.info(f"Connecting to {self.URI}")
            self.client = MilvusClient(
                uri=self.URI,
                token=f"{self.Zilliz_CLUSTER_USER}:{self.Zilliz_CLUSTER_PWD}",
            )
            #creating schema
            self.schema = self.client.create_schema(
                auto_id = False,
                enable_dynamic_field=True
            )
            self.schema.add_field(field_name="id",datatype=DataType.INT64, is_primary = True, description="primary id")
            self.schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=self.vector_field_dim, description="vector")
            self.schema.add_field(field_name="text",datatype=DataType.VARCHAR,max_length=65535, description="text content")
            self.index_params = self.client.prepare_index_params()
            self.index_params.add_index(
                field_name="id",
                index_type="STL_SORT"
            )
            self.index_params.add_index(
                field_name="vector",
                index_type="IVF_FLAT", #Quantization-based index; High-speed query & Requires a recall rate as high as possible
                index_name="vector_index",
                metric_type=self.metric_type, #inner product
                params={"nlist": 128 } #IVF_FLAT divides vector data into nlist cluster units; Range: [1, 65536]; default value: 128
            )
        except Exception as ex:
            logger.error(f"[Error] Unable to connect to Milvus Client: {ex}")
            raise Exception(f"[Error] Unable to connect to Milvus Client: {ex}")
    
    def _listout_collections(self):
        """List all collections in the database"""
        try:
            collections = utility.list_collections()
            logger.info(f"List of available collections: {collections}")
        except MilvusException as e:
            logger.error(f"Failed to list collections: {e}")
            raise Exception(f"Failed to list collections: {e}")

    def _create_collection(self):
        """Create new collection or load existing one"""
        try:
            if not self.client.has_collection(collection_name=self.collection_name):
                logger.info(f"Creating collection: {self.collection_name}; vector dimension: {self.vector_field_dim}")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    schema=self.schema,
                    consistency_level="Strong",
                )
                # Create index for new collection
                self.client.create_index(
                    collection_name=self.collection_name,
                    index_params=self.index_params
                )
                logger.info(f"Loading {self.collection_name} collection...!!!")
                self.client.load_collection(
                    collection_name=self.collection_name,
                    replica_number=1  # Adjust based on your needs
                )
                # 
            else:
                logger.info(f"Collection {self.collection_name} already exists. Loading it...!!!")
                # Load the collection (whether new or existing)
                self.client.load_collection(
                    collection_name=self.collection_name,
                    replica_number=1  # Adjust based on your needs
                )
                # Wait for collection to be loaded
                load_state = self.client.get_load_state(
                    collection_name=self.collection_name,
                    partition_name="",
                    timeout=None
                ) # Use default timeout
                logger.info(f"Load state: {load_state}")
                if not load_state:
                    raise Exception(f"Collection {self.collection_name} failed to load - empty state returned")
        except MilvusException as e:
            raise Exception(f"[Error] Failed to create/load {self.collection_name}:{e}")

    def _insert_data(self, data):
        """Inserting data into collection"""
        logger.info("Inserting embedded data into Milvus server")
        try:
            if data and any(item["id"] is not None for item in data):
                self.client.insert(
                    collection_name=self.collection_name,  
                    data=data
                )
                logger.info(f"Successfully inserted {len(data)} embedded data into {self.collection_name}")
            else:
                logger.error("[Error] No valid data to insert. Skipping insertion.")
        except Exception as ex:
            logger.error(f"[Error] Unable to insert data to Milvus:{ex}")
            raise Exception(f"[Error] Unable to insert data to Milvus: {ex}")

    def _search_and_output_query(self, query_embeddings: List, response_limit: int = 3,  json_indent:int = 3):
        """Generates Embeddings and returns retrieved data
        Args:
            query_embeddings (List): List of user query embeddings
            response_limit (int): Number of output responses from database
            json_indent (int): Indentation limit for Json output format
        Raises:
            Exception: If unable to retrieve data
        """
        try:
            logger.info(f"[Important] using: {self.metric_type} metric type")
            search_res = self.client.search(
            collection_name=self.collection_name,
            anns_field="vector",
            data=[query_embeddings[0]],  
            limit= response_limit,  # Return top 5 results
            search_params={"metric_type": self.metric_type,  "params": {}},  # Inner product distance
            output_fields=["text"],  # Return the text field
            )
        except Exception as e:
            logger.error(f"[Error] unable to query search: {e}")
            raise Exception(f"[Error] unable to query search: {e}")
        output = self._get_retrieved_info(search_res=search_res,json_indent=json_indent)
        return output