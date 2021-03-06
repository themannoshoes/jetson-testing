#include "mj_store_test.h"
#include "use_a_prtcol_to_pack_store.h"


#define GET_DATA_FROM_BUF(data,src_buf,flag)  convert_the_data_endian((uint8_t *)&data, (uint8_t*)src_buf, sizeof(data), flag)


store_type store_util;
data_info temp_one_data[3 * 4];
float test_src_data2 = 0.6;
uint8_t test_src_data1 = 23;
int16_t test_src_data3 = -2;
float test_src_data4 = 0.9;


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
       default:
          return 0;
    }
}

/* run this function will only stroe one pack of data in the table

*/
void init_store_sturcture(store_type * store_p, uint8_t * buf_head_addr, uint8_t *buf_end_addr)
{

    store_p->current_read_cursor  = buf_head_addr;
    store_p->current_write_cursor = buf_head_addr;
    store_p->store_mem_head_p     = buf_head_addr;
    store_p->store_mem_tail_p     = buf_end_addr;
 
    store_p->data_num_in_unit = 0;
    store_p->unit_table = NULL;
    store_p->unit_data_dim    = 0;
    store_p->start_flag = 0;
}
/*
    Description: 
            due to the data serial number in table,we get the name id ,then sample a specified data, and then store it in "one_data"
    para1:  table pointer
    para2:  data serial number in the table
    para3:  a pointer to data buffer
*/
void sample_one_data_by_table(uint8_t * table_p, uint16_t data_snum, data_info * one_data)
{
    uint16_t table_data_amount = *(__packed uint16_t *)(table_p);
    uint32_t data_name_id_offset = data_snum * 2 + 4;
 
    uint8_t * temp_table_cursor = NULL;
 
    if(data_name_id_offset > (table_data_amount -1) * sizeof(uint16_t) + 4){ //the offset is exceed the table 
        return;
    }
    temp_table_cursor = table_p + data_name_id_offset;
    one_data->data_name_id =  *(__packed uint16_t*)(temp_table_cursor);

    switch(one_data->data_name_id){
        default:
        case 0:
            one_data->_8_ = test_src_data1;
            one_data->data_type_id = ST_ID_INT8;
            one_data->data_size = get_size_of_datatype(one_data->data_type_id);
        break;
        case 1:
            one_data->_f_ = test_src_data2;
            one_data->data_type_id = ST_ID_FLOAT;
            one_data->data_size = get_size_of_datatype(one_data->data_type_id);
        break;
        case 2:
            one_data->_16_ = test_src_data3;
            one_data->data_type_id = ST_ID_INT16;
            one_data->data_size = get_size_of_datatype(one_data->data_type_id);
        break;
        case 3:
            one_data->_f_ = test_src_data4;
            one_data->data_type_id = ST_ID_FLOAT;
            one_data->data_size = get_size_of_datatype(one_data->data_type_id);        
        break;
    }
}
/*
Description:
    this function will return a offset 
para1:
para2:
para3:  0: function will return an normal offset
        1: function will return an offset that add the size of last data 
return:  a offset of the data in the buffer
*/
uint16_t get_data_offset_in_the_buffer(uint8_t * table_p, uint16_t data_snum_in_table, uint8_t type_ret)
{

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
    if(type_ret == 0 ){
        return temp_size - one_data.data_size;
    }else{
        return temp_size;
    }
}

/*
Description:
  this function will return one data unit dimension that timestamp not included
para1:
return: the unit dimension that timestamp not included
*/
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

uint16_t calculate_table_dim(uint8_t * table_p)
{
    uint16_t table_dim = 0;
    uint16_t data_amount = 0;
 
    data_amount = *(__packed uint16_t*)(table_p + 0);
    table_dim = data_amount * sizeof(uint16_t) + 4;
    return table_dim;
}

/* Description:
          run this function, we would store a unit data into memory, unit data is from realtime sampling
   para1: store_entity pointer
   para2: a table pointer;
          the data needed to be stored is indicated by this table
   para3: timestamp. it will be store with the data_unit
*/
void sample_n_store_data_unit_in_mem(store_type * store_p, uint8_t * table_p, uint32_t timestamp)
{
   uint8_t *buf_w_cursor = (uint8_t *)(store_p->current_write_cursor);
 
   uint16_t data_cur_snum = 0;
   uint16_t table_dim = 0;
   data_info one_data;
 
   if(store_p->start_flag == 0){
      table_dim = calculate_table_dim(table_p);
      if(store_p->unit_table != NULL){
          free((void *)store_p->unit_table);
          store_p->unit_table = NULL;
      }
      store_p->data_num_in_unit = *(__packed uint16_t *)(table_p + 0);
      store_p->unit_data_dim = calculate_data_unit_dim(table_p);
      store_p->unit_table = malloc(table_dim);
      if(store_p == NULL){
          return;
      }
      memmove((void *)store_p->unit_table, (void *)table_p, table_dim);
      
      store_p->start_flag = 1;    //when evrything init done, set the start flag
   }
   
   if( (uint8_t *)store_p->current_write_cursor + store_p->unit_data_dim + 4 > store_p->store_mem_tail_p + 1){ //prevent overwrite
       return;
   }
   
   while(data_cur_snum < store_p->data_num_in_unit){   /*store the whole table data*/
    
       sample_one_data_by_table(table_p, data_cur_snum, &one_data);
       switch(one_data.data_type_id){
        default:
        case ST_ID_UINT8:
        case ST_ID_INT8:
            *(__packed uint8_t *)(buf_w_cursor) = one_data._8_;
            buf_w_cursor = buf_w_cursor + sizeof(one_data._8_);
        break;
        case ST_ID_UINT16:
        case ST_ID_INT16:
            *(__packed uint16_t *)(buf_w_cursor)= one_data._16_;
            buf_w_cursor = buf_w_cursor + sizeof(one_data._16_);
          break;
        case ST_ID_UINT32:
        case ST_ID_INT32:
            *(__packed uint32_t *)(buf_w_cursor) = one_data._32_;
            buf_w_cursor = buf_w_cursor + sizeof(one_data._32_);
          break;
        case ST_ID_UINT64:
        case ST_ID_INT64:
            *(__packed uint64_t *)(buf_w_cursor) = one_data._64_;
            buf_w_cursor = buf_w_cursor + sizeof(one_data._64_);
          break;
        case ST_ID_FLOAT:
            *(__packed float *)(buf_w_cursor) = one_data._f_;
            buf_w_cursor = buf_w_cursor + sizeof(one_data._f_);
          break;
        case ST_ID_DOUBLE:
            *(__packed double *)(buf_w_cursor) = one_data._db_;
            buf_w_cursor = buf_w_cursor + sizeof(one_data._db_);
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


/*
Description:
    run this function, we can get one data from one unit in the memory
para1
para2:  the table pointer
para3:  the pointer to one unit
para4:  the data serial number in the table
pata5(rw):  the pointer using to save one_data
return value: 
*/
int8_t seek_one_data_in_store_test(store_type * store_p,uint8_t * table_p, uint8_t * head_cursor_to_one_unit, uint16_t data_snum_of_unit, data_info * one_data)
{
    uint16_t data_amount = 0;
    data_amount = *(uint16_t *)(table_p + 0);
    uint16_t offset;

    if(store_p->start_flag == 0){
        return -1; //we didnot store anything
    }
    if(data_snum_of_unit < data_amount){

        sample_one_data_by_table(table_p, data_snum_of_unit, one_data);
        offset = get_data_offset_in_the_buffer(table_p, data_snum_of_unit,0);
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
    }else if(data_snum_of_unit == data_amount){
     
        offset = get_data_offset_in_the_buffer(table_p, data_snum_of_unit-1,1);
        one_data->data_name_id = 0;//fake, no use 
        one_data->data_size    = sizeof(uint32_t);
        one_data->data_type_id = ST_ID_UINT32;
        one_data->_32_ = *(__packed int32_t *)(head_cursor_to_one_unit + offset);
    }else{
        return -1; //error
    }
    return 0;

}

/*
Description:
    run this function, we can get one data(Or timestamp) from one unit in the memory
para1:  the table pointer
para2:  the pointer to one unit
para3:  the data serial number in the table

return value: if success, return the datasize that we seek
*/
int8_t seek_one_data_in_store_N_copy_2_buf(store_type * store_p,uint8_t * table_p, uint8_t * head_cursor_to_one_unit, uint16_t data_snum_of_unit, uint8_t * buffer_pnter)
{
    uint16_t data_amount = 0;
    data_info one_data_bucket;
    data_amount = *(uint16_t *)(table_p + 0);
    uint16_t offset = 0;
 
    if(store_p->start_flag == 0){
        return -1; //we didnot store anything
    }
    if(data_snum_of_unit <  data_amount){
        sample_one_data_by_table(table_p, data_snum_of_unit, &one_data_bucket);
        offset = get_data_offset_in_the_buffer(table_p, data_snum_of_unit,0);
        switch(one_data_bucket.data_type_id){
            default:
            case ST_ID_INT8:
            case ST_ID_UINT8:
                *(int8_t *)buffer_pnter = *(__packed int8_t *)(head_cursor_to_one_unit + offset);
            break;
            case ST_ID_UINT16:
            case ST_ID_INT16:
                *(int16_t *)buffer_pnter = *(__packed int16_t *)(head_cursor_to_one_unit + offset);
            break;
            case ST_ID_UINT32:
            case ST_ID_INT32:
                *(int32_t *)buffer_pnter = *(__packed int32_t *)(head_cursor_to_one_unit + offset);
            break;
            case ST_ID_UINT64:
            case ST_ID_INT64:
                *(int64_t *)buffer_pnter = *(__packed int64_t *)(head_cursor_to_one_unit + offset);
            break;
            case ST_ID_FLOAT:
                *(float *)buffer_pnter = *(__packed float *)(head_cursor_to_one_unit + offset);
            break;
            case ST_ID_DOUBLE:
                *(double *)buffer_pnter = *(__packed double *)(head_cursor_to_one_unit + offset);
            break;
        }
        return one_data_bucket.data_size;
    }else if(data_snum_of_unit == data_amount){ //in this case, we copy the timestamp into buffer
        offset = get_data_offset_in_the_buffer(table_p, data_snum_of_unit -1,1);
        *(uint32_t *)buffer_pnter = *(__packed uint32_t *)(head_cursor_to_one_unit + offset);
        return sizeof(uint32_t);
    }else{
        return -1; //error
    }
}

/*
  Description:
     in this function, we would store a unit data (timestamp included) into a buffer 
  para1:
  para2:
  para3:
  para4(w): "unit_buffer"  the buffer to save the data we get
  return: -1:error;
*/
int8_t seek_one_unit_data_in_store_test(store_type * store_p, uint8_t * table_p, uint8_t * head_cursor_to_one_unit, uint8_t * unit_buffer, uint16_t buffer_size)
{
    //get every data into a buffer
    uint16_t data_snum = 0;
    int8_t ret_num_of_seek;
    uint16_t unit_buffer_cursor = 0;
    uint16_t unit_dim;
    
    unit_dim = calculate_data_unit_dim(table_p);
    unit_dim += 4; //add timestamp
    if(unit_dim > buffer_size){
        return -1;
    }
    
    while(data_snum <= store_p->data_num_in_unit){
        
        ret_num_of_seek = seek_one_data_in_store_N_copy_2_buf(store_p ,table_p, head_cursor_to_one_unit, data_snum, unit_buffer + unit_buffer_cursor);
        if(ret_num_of_seek == -1){
            return -1;
        }
        unit_buffer_cursor += ret_num_of_seek;
        data_snum++;
    }
    return 0;
}
/*
Description: 
    get data units into buffer, as more as we can
para1:
para2:
para3:
para4(w):  "unit_buffer", the buffer that save the data we get
return :   return the size that we get
*/
int32_t get_data_units_into_buffer(store_type * store_p, uint8_t * table_p, uint8_t * head_cursor_to_one_unit, uint8_t * unit_buffer, uint16_t buffer_size)
{
    uint8_t i;
    int8_t ret = 0;
    uint16_t how_many_units_can_pack = 0;
    uint16_t dims_of_units = 0;
    
    dims_of_units = calculate_data_unit_dim(table_p);
    dims_of_units +=4;
    how_many_units_can_pack = buffer_size / dims_of_units;
    if(how_many_units_can_pack == 0){
        return -1; //error
    }
    
    for(i = 0;i < how_many_units_can_pack;i++){
        ret = seek_one_unit_data_in_store_test(store_p, table_p, head_cursor_to_one_unit + i*dims_of_units, unit_buffer + i*dims_of_units, dims_of_units);
        if(ret == -1)return -1;
    }
    return dims_of_units * how_many_units_can_pack;
}

/*
Description:
    pack the data in the store mem, as more as we can
para1:
para2:
para3:
return: 0 : OK
*/
int8_t pack_the_data_not_read(store_type * store_p, uint8_t * packer_buffer, uint16_t buffer_size)
{
    int32_t ret;
    
    if(store_p->start_flag == 0)return -1;
 
    if(store_p->current_read_cursor + buffer_size > store_p->store_mem_tail_p
    || store_p->current_read_cursor < store_p->store_mem_head_p
    || store_p->current_read_cursor + buffer_size > store_p->current_write_cursor){
        return -1; //error
    }
    ret = get_data_units_into_buffer(store_p, store_p->unit_table, store_p->current_read_cursor, packer_buffer, buffer_size);
    if(ret == -1)return -1;//error
 
    store_p->current_read_cursor = store_p->current_read_cursor + ret;
    return 0;
}

void delete_the_store_entity(store_type * store_p){
    
    if(store_p->unit_table != NULL){
        free((void *)store_p->unit_table);
    }
}

/****************************************************************************************/
/*test code*/



void test_store_code(uint32_t timestamp)
{
    uint8_t store_count = 3;
    
    store_type store_entity;
    uint8_t * temp_buffer;
    uint16_t unit_N_tstamp_size;
    
    uint8_t * table_p = malloc(4 + 4*sizeof(int16_t));;
 
    *(__packed uint16_t *)(table_p + 0) = 4; //total num
    *(__packed uint16_t *)(table_p + 2) = 1; //version id
    *(__packed uint16_t *)(table_p + 4) = 0; //data id 
    *(__packed uint16_t *)(table_p + 6) = 1; //data id
    *(__packed uint16_t *)(table_p + 8) = 2; //data id
    *(__packed uint16_t *)(table_p + 10) = 3; //data id

    unit_N_tstamp_size = calculate_data_unit_dim(table_p);
    unit_N_tstamp_size += 4;
    temp_buffer  = malloc(3*unit_N_tstamp_size);
 
    init_store_sturcture(&store_entity, (uint8_t *)BUFFER_HEAD_ADDR, (uint8_t *)BUFFER_END_ADDR);

    while(store_count > 0){
        sample_n_store_data_unit_in_mem(&store_entity, table_p, 0x3344); 
        test_src_data1--;
        test_src_data2--;
        test_src_data3--;
        test_src_data4--;
        store_count --;
    }
    
//    get_data_units_into_buffer(&store_entity, 
//                                             table_p,
//                                             store_entity.current_read_cursor,
//                                             temp_buffer,
//                                             3*unit_N_tstamp_size);
    
//    for(uint8_t j =0; j < 3; j++){
//        for(uint8_t i = 0;i < 4; i++){
//            seek_one_data_in_store_test(&store_entity,table_p, temp_buffer + j*unit_N_tstamp_size, i, &temp_one_data[i+j * 4]);
//        }
//    }

    for(uint8_t i = 0;i < 2;i++){
        g_packer._tx_id         = 0x99;
        g_packer._tx_sub_id     = 0x01;
        g_packer._rx_id         = 0x99;
        g_packer._rx_sub_id     = 0x01;
        g_packer._frm_type      = 0xaa;
        pack_the_data_not_read(&store_entity, g_packer._data._base, 2*unit_N_tstamp_size);
        packer_run(&g_packer, g_packer._data._base, 2*unit_N_tstamp_size, NULL);
        packer_download_by_dlink1(&g_packer, NULL);
        HAL_Delay(1000);
        
        test_parse_the_packer(g_packer._frame._base,g_packer._frame._size,&store_entity);
    }

    free(table_p);
    free(temp_buffer);
    delete_the_store_entity(&store_entity);

}



/******************************************************below is for simple endian ******************************************************/
/*
return value: 1 means this is big endian, 0 means this is small endian
*/
uint8_t check_system_endian()
{
    uint16_t test_num = 0x1798;
    uint8_t* p_num = (uint8_t *)&test_num;
 
    if(*p_num == 0x17
    && *(p_num + 1) == 0x98){
        return 0; //small endian
    }else if(*p_num == 0x98
    && *(p_num + 1) == 0x17){
        return 1; //big endian
    }else{
        return 2; //we don't know what it is
    }
}
/*para4: 1 means raw copy, without convert */
void convert_the_data_endian( uint8_t * dst_pointer, uint8_t *src_pointer, uint8_t datasize, uint8_t flag_endian)
{
    uint8_t i;
    switch(flag_endian){
        case 0:
            for(i = 0;i < datasize; i++){
                    memmove( dst_pointer + i , src_pointer + datasize-1 -i,1);
            }
        break;
        case 1:
            memmove(dst_pointer,src_pointer, datasize);
        break;
    }
}

float test_float = 123.586;
uint32_t test_32 = 0x12345678;
float temp_result = 0;
uint32_t result_32 = 0;
uint8_t flag_of_endian = 0;
void test_endian_transf()
{

    uint8_t bucket[8];
    uint8_t bucket_1[8];
    uint8_t* buffer = &bucket[0];
    uint8_t * buffer_1 = &bucket_1[0];
 
    flag_of_endian = check_system_endian();

    convert_the_data_endian(buffer, (uint8_t *)&test_float, 4,0);
    
    GET_DATA_FROM_BUF(temp_result, buffer, 0);
 
    convert_the_data_endian(buffer, (uint8_t *)&test_32, 4,0);
 
    GET_DATA_FROM_BUF(result_32, buffer, 0);
}




void assemble_a_data_from_buffer(uint8_t *buffer, uint8_t *result_p, uint8_t datasize)
{
    

}
