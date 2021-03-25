#include "mj_store_test.h"

#define BUFFER_HEAD_ADDR 0xC0000000
#define BUFFER_END_ADDR  0xC0800000


#define FRAME_BUF_SIZE  250

typedef enum _STORE_DATA_TYPE_ID{
    ST_ID_UINT8,
    ST_ID_UINT16,
    ST_ID_UINT32,
    ST_ID_UINT64,
    ST_ID_INT8,
    ST_ID_INT16,
    ST_ID_INT32,
    ST_ID_INT64,
    ST_ID_FLOAT,
    ST_ID_DOUBLE
}S_TYPE_ID;

typedef struct _data_information{
    uint16_t   data_name_id;
    S_TYPE_ID  data_type_id;
    uint8_t    data_size;
    uint8_t  _8_;
    uint16_t _16_;
    uint32_t _32_;
    uint64_t _64_;
    float    _f_;
    double   _db_;
    
}data_info;

typedef struct _store_structure{
    uint8_t * current_write_cursor;    //the current buffer we can write
    uint8_t * current_read_cursor;  //the reading cursor when we read out data of store_memory, it point to the current buffer we haven't read data out yet
    uint8_t * store_mem_head_p; //the head pointer, it point to the head buffer that store the first data
    uint8_t * store_mem_tail_p; //the tail pointer, it point to the tail buffer that store the last data
    uint8_t * unit_table;
 
    uint16_t data_num_in_unit;    //the num of an unit data in memory
    uint16_t unit_data_dim;      //the dimension of an unit data in memory
    uint8_t start_flag;
}store_type;



store_type store_util;
data_info temp_one_data[3];

void sendout_table()
{
    //just send out a uint16_t to GCS, then GCS would know the total number of the table
    //an a table_version value
    //the table is static, once the code is compiled, table would not be changed
}

uint8_t get_size_of_datatype(S_TYPE_ID datatype_id)
{
    switch(datatype_id){
       case ST_ID_UINT8: //uint8_t 
       case ST_ID_INT8: //int8_t 
          return 1;
       case ST_ID_UINT16: //uint16_t
       case ST_ID_INT16: //int16_t
          return 2;
       case ST_ID_UINT32: //uint32_t 
       case ST_ID_INT32: //int32_t 
       case ST_ID_FLOAT: //float 
          return 4;
       case ST_ID_UINT64: //uint64_t 
       case ST_ID_INT64: //int64_t
       case ST_ID_DOUBLE: //double 
          return 8;
    }
}

/* run this function will only stroe one pack of data in the table

*/
void init_store_sturcture(store_type * store_p){
    store_p->start_flag = 0;
    store_p->current_read_cursor  = (uint8_t *)BUFFER_HEAD_ADDR;
    store_p->current_write_cursor = (uint8_t *)BUFFER_HEAD_ADDR;
    store_p->store_mem_head_p = (uint8_t *)BUFFER_HEAD_ADDR;
    store_p->store_mem_tail_p = (uint8_t *)BUFFER_END_ADDR;
 
    store_p->data_num_in_unit = 0;
    store_p->unit_data_dim    = 0;
}

void sample_one_data_by_table(uint8_t * table_p, uint16_t data_snum, data_info * one_data)
{
    uint16_t table_size = *(__packed uint16_t *)(table_p);
    uint32_t data_name_id_offset = data_snum * 2 + 4;
 
    uint8_t * temp_table_cursor = NULL;
 
    if(data_name_id_offset > table_size - 1){
        return;
    }
    temp_table_cursor = table_p + data_name_id_offset;
    one_data->data_name_id =  *(__packed uint16_t*)(temp_table_cursor);

    switch(one_data->data_name_id){
        default:
        case 0:
            one_data->_8_ = 255;
            one_data->data_type_id = ST_ID_INT8;
            one_data->data_size = get_size_of_datatype(one_data->data_type_id);
        break;
        case 1:
            one_data->_f_ = 0.6;
            one_data->data_type_id = ST_ID_FLOAT;
            one_data->data_size = get_size_of_datatype(one_data->data_type_id);
        break;
        case 2:
            one_data->_16_ = -2;
            one_data->data_type_id = ST_ID_INT16;
            one_data->data_size = get_size_of_datatype(one_data->data_type_id);
        break;
    }
}

uint16_t get_data_offset_in_the_buffer(uint8_t * table_p, uint16_t data_snum_in_table){ //data_snum_in_table: start from zero

    data_info one_data;
    uint16_t count_up = 0;
    uint16_t temp_size = 0;
    
    uint16_t total_data_num = *(__packed uint16_t*)table_p;
    if(data_snum_in_table > total_data_num -1){
        return 0;
    }

    while(count_up <= data_snum_in_table){
        sample_one_data_by_table(table_p, count_up, &one_data);
        temp_size += one_data.data_size; 
        count_up ++;
    }
    return temp_size;
}

uint16_t calculate_data_unit_dim(void * table_p)
{
    uint16_t data_cur_snum = 0;
    uint16_t temp_dim = 0;
    data_info temp_one_data;

    uint16_t data_total_num = *(__packed uint16_t *)table_p;
 
    while(data_cur_snum < data_total_num){
        sample_one_data_by_table(table_p, data_cur_snum, &temp_one_data);
        temp_dim +=  get_size_of_datatype(temp_one_data.data_type_id);
        data_cur_snum++;
    }
    return temp_dim;
}

/* run this function:  we would store a unit data into memory, unit data is from realtime sampling*/
void store_data_unit_in_mem(store_type * store_p, uint8_t * table_p, uint32_t timestamp) //para need: buffer pointer to store;  table pointer; 
{
   uint8_t *buf_w_cursor = (uint8_t *)(store_p->current_write_cursor);
 
   uint16_t data_cur_snum = 0; //start from zero
   data_info one_data;
 
   if(store_p->start_flag == 0){
      store_p->start_flag = 1;
      store_p->data_num_in_unit = *(__packed uint16_t *)table_p;
      store_p->unit_data_dim = calculate_data_unit_dim(table_p);
      memmove((void *)store_p->unit_table, (void *)table_p, store_p->data_num_in_unit *2 + 4); 
   }
   if( (uint8_t *)store_p->current_write_cursor + store_p->unit_data_dim + 4 > store_p->store_mem_tail_p + 1){
       return;
   }
   
   while(data_cur_snum < store_p->data_num_in_unit){   /*store the whole table data*/
    
       sample_one_data_by_table(table_p, data_cur_snum, &one_data);
       switch(one_data.data_type_id){
        default:
        case ST_ID_UINT8:
        case ST_ID_INT8:
            *(__packed uint8_t *)(buf_w_cursor) = one_data._8_;
            buf_w_cursor = buf_w_cursor + 1;
        break;
        case ST_ID_UINT16:
        case ST_ID_INT16:
            *(__packed uint16_t *)(buf_w_cursor)= one_data._16_;
            buf_w_cursor = buf_w_cursor + 2;
          break;
        case ST_ID_UINT32:
        case ST_ID_INT32:
            *(__packed uint32_t *)(buf_w_cursor) = one_data._32_;
            buf_w_cursor = buf_w_cursor + 4;
          break;
        case ST_ID_UINT64:
        case ST_ID_INT64:
            *(__packed uint64_t *)(buf_w_cursor) = one_data._64_;
            buf_w_cursor = buf_w_cursor + 8;
          break;
        case ST_ID_FLOAT:
            *(__packed float *)(buf_w_cursor) = one_data._f_;
            buf_w_cursor = buf_w_cursor + 4;
          break;
        case ST_ID_DOUBLE:
            *(__packed double *)(buf_w_cursor) = one_data._db_;
            buf_w_cursor = buf_w_cursor + 8;
        break;
       }
       data_cur_snum ++;
   }

   //store the timestamp
   *(__packed uint32_t *)(buf_w_cursor) = timestamp;
   buf_w_cursor = buf_w_cursor + 4;
   
   //update the global current wrtite cursor
   store_p->current_write_cursor = (void *)buf_w_cursor;
}

void pack_most_data_in_a_frame(store_type * store_p, packer_t * gp_packer, uint32_t timestamp){  //frame_buf is the data_pointer for a airship frame
 
    uint8_t * cur_r_cursor = store_p->current_read_cursor;
    uint16_t data_unit_size = store_p->data_num_in_unit;
    uint8_t how_many_unit_can_store = FRAME_BUF_SIZE / data_unit_size;
 
    uint16_t data_size_in_a_frm = how_many_unit_can_store * data_unit_size;
    if(data_size_in_a_frm > FRAME_BUF_SIZE){
        return;
    }

    //move data into frame buf 
    memmove((void *)gp_packer->_data._base, (const void *)cur_r_cursor, data_size_in_a_frm);
    gp_packer->_timestamp     = timestamp;
    gp_packer->_tx_id         = AIRSHIP_SHIP_ID;
    gp_packer->_tx_sub_id     = AIRSHIP_FLY_CTL_BOARD_ID;
    gp_packer->_rx_id         = AIRSHIP_GCS_ID;
    gp_packer->_rx_sub_id     = AIRSHIP_FLY_CTL_BOARD_ID;
    gp_packer->_frm_type      = AIRSHIP_FRAME_TYPE_STORE_TABLE_TESTING;

    packer_run(gp_packer, gp_packer->_data._base, data_size_in_a_frm, NULL);
}

void seek_one_data_in_store_test(uint8_t * table_p, uint8_t * head_cursor_to_one_unit, uint16_t data_snum_of_unit, data_info * one_data) //data_num_of_first_unit: start from zero
{
    sample_one_data_by_table(table_p, data_snum_of_unit, one_data);
    
    uint16_t offset = get_data_offset_in_the_buffer(table_p, data_snum_of_unit);

    switch(one_data->data_type_id){
        default:
        case ST_ID_INT8:
        case ST_ID_UINT8:
            one_data->_8_ = *(__packed int8_t *)(head_cursor_to_one_unit + offset);
        break;
        case ST_ID_UINT16:
        case ST_ID_INT16:
            one_data->_16_ = *(__packed int16_t *)(head_cursor_to_one_unit + offset);
        break;
        case ST_ID_UINT32:
        case ST_ID_INT32:
            one_data->_32_ = *(__packed int32_t *)(head_cursor_to_one_unit + offset);
        break;
        case ST_ID_UINT64:
        case ST_ID_INT64:
            one_data->_64_ = *(__packed int64_t *)(head_cursor_to_one_unit + offset);
        break;
        case ST_ID_FLOAT:
            one_data->_f_ = *(__packed float *)(head_cursor_to_one_unit + offset);
        break;
        case ST_ID_DOUBLE:
            one_data->_db_ = *(__packed double *)(head_cursor_to_one_unit + offset);
        break;
    }
}

void parse_data_in_frame_buf(void * table_p, uint8_t * frame_buf, uint8_t unit_amount){
   
    uint8_t i,j;
    uint16_t temp_unit_dim = calculate_data_unit_dim(table_p);
    for(i = 0;i < unit_amount;i++){
        
         for(j = 0; j< temp_unit_dim; j++){
             
         }
    }
}

void test_store_code(uint32_t timestamp)
{
    uint8_t store_count = 3;
    
    store_type store_entity;
    store_entity.unit_table = malloc(4 + 3*2);
    if(store_entity.unit_table == NULL){
        return;
    }
    uint8_t * table_p = store_entity.unit_table;
 
    *(__packed uint16_t *)(table_p + 0) = 3; //total num
    *(__packed uint16_t *)(table_p + 2) = 1; //version id
    *(__packed uint16_t *)(table_p + 4) = 0;
    *(__packed uint16_t *)(table_p + 6) = 1;
    *(__packed uint16_t *)(table_p + 8) = 2;

    init_store_sturcture(&store_entity);

    while(store_count > 0){
        store_data_unit_in_mem(&store_entity, table_p, timestamp); 
        store_count --;
    }

    for(uint8_t i = 0;i < 3; i++){
        seek_one_data_in_store_test(table_p, store_entity.current_read_cursor, i, &temp_one_data[i]);
    }
}
