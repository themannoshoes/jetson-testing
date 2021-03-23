#include <stdint.h>

typedef struct _data_information{
    uint16_t data_name_id;
    uint8_t  data_type;
    uint8_t  _8_;
    uint16_t _16_;
    uint32_t _32_;
    uint64_t _64_;
    float    _f_;
    double   _db_;
    
}data_info;

typedef _store_structure{
    void * current_write_cursor;
 
}store_type;

store_type store_util;

void sendout_table()
{
    //just send out a uint16_t to GCS, then GCS would know the total number of the table
    //an a table_version value
    //the table is static, once the code is compiled, table would not be changed

}
void parse_table_back(void * src_table, uint16_t size_of_table)
{

    static uint16_t total_data_amount = 0;
    uint8_t * r_cursor = (uint8_t *)src_table;
    uint8_t data_type;
    
 
    total_data_amount = *(__packed uint16_t *)(r_cursor + 0)
    while(){
        data_type = *(__packed uint8_t *)(r_cursor + i);
        switch(data_type){
         case 0: //this case is uint8_t 
            
            break;
         case 1: //this case is uint16_t
            break;
         case 2: //this case is uint32_t 
            break;
         case 3: //uint64_t 
            break;
         case 4: //int8_t 
            break;
         case 5: //int16_t
            break;
         case 6: //int32_t 
            break;
         case 7: //int64_t 
            break;
         case 8: //float 
            break;
         case 9: //double 
            break;
        }
     
        
        i = i + 3;
    }
}

/* run this function will only stroe one pack of data in the table

*/

void init_stroing_function(){
 
    table_cursor = 0;
}

void get_one_table_data(void * table_p, uint16_t data_snum, data_info * one_data)
{
    uint32_t data_offset = data_snum * 2 + 4;
 
    one_data->data_name_id =  *(__packed uint16_t*)(table_p + data_offset);

    switch(one_data->data_name_id){
        default:
        case 0:
            one_data->_8_;
            one_data->data_type = 4;
        break;
        case 1:
            one_data->_f_;
            one_data->data_type = 8;
        break;
        case 2:
            one_data->_16_;
            one_data->data_type = 5;
        break;
    }
    
}

/* pseducode*/
void store_one_pseducode(store_type * store_p,void * table_p, void * buffer_p, uint32_t timestamp) //para need: buffer pointer to store;  table pointer; 
{
   //data needed:  data_curnum
   void *buf_w_cursor = store_p->current_write_cursor;
   uint16_t total_num = *(__packed uint16_t *)table_p;
   uint16_t data_cur_snum = 0; //start from zero

   //store the whole table data
   while(data_cur_snum < total_num){
    
       get_one_table_data(void * table_p, uint16_t data_cur_snum, data_info * one_data);
    
       switch(one_data.data_name_id){
        case 0:
        case 4:
            *(__packed uint8 *)(buffer_p + buf_w_cursor) = one_data->_8_;
            buf_w_cursor = buf_w_cursor + 1;
        break;
        case 1:
        case 5;
            *(__packed uint16 *)(buffer_p + buf_w_cursor)= one_data->_16_;
            buf_w_cursor = buf_w_cursor + 2;
          break;
        case 2:
        case 6
            *(__packed uint32 *)(buffer_p + buf_w_cursor) = one_data->_32_;
            buf_w_cursor = buf_w_cursor + 4;
          break;
        case 3:
        case 7
            *(__packed uint64 *)(buffer_p + buf_w_cursor) = one_data->_64_;
            buf_w_cursor = buf_w_cursor + 8;
          break;
        case 8:
            *(__packed float *)(buffer_p + buf_w_cursor) = one_data->_f_;
            buf_w_cursor = buf_w_cursor + 4;
          break;
        case 9:
            *(__packed double *)(buffer_p + buf_w_cursor) = one_data->_db_;
            buf_w_cursor = buf_w_cursor + 8;
        break;
       }
       data_cur_snum ++;
   }
   
   //store the timestamp
   *(__packed uint32_t *)(buffer_p + buf_w_cursor) = timestamp;
   buf_w_cursor = buf_w_cursor + 4£»
 


}



void sending_out_data(int table, uint16_t size); void get_most_datainto_one_frame();
void parsing_all_data(); void data_parse_one_frame();
void package_divide
void package_parser();
void package_d
 