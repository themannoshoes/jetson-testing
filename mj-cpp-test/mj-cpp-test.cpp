#include <iostream>



#define ITEM 5
int main()
{
    int intArray[ITEM] = {1,2,3,4,5};
    char charArray[ITEM] = {'F', 'i','s','h','C'};

    int *intPtr = intArray;
    char *charPtr = charArray;

    std::cout << "type int array output : " << '\n';
    for(int i = 0; i < ITEM; i++){
        std::cout << *intPtr << "at" << reinterpret_cast<unsigned long>(intPtr) << '\n';
        std::cout << *intPtr << "at" << intPtr << '\n';
        std::cout << *intPtr << "at" << (unsigned long)intPtr << '\n';
        intPtr++;
    }

    std::cout << "type char array output :" << '\n';
    for(int i =0;i < ITEM;i++){
        std::cout << *charPtr << "at" << reinterpret_cast<unsigned long>(charPtr) << '\n';
        std::cout << *intPtr << "at" << charPtr << '\n';
        std::cout << *intPtr << "at" << (unsigned long)charPtr << '\n';
        charPtr++;
    }
    return 0;



}