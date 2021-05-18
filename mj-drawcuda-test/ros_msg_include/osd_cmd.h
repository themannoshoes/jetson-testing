// Generated by gencpp from file learning_topic/osd_cmd.msg
// DO NOT EDIT!


#ifndef LEARNING_TOPIC_MESSAGE_OSD_CMD_H
#define LEARNING_TOPIC_MESSAGE_OSD_CMD_H


#include <string>
#include <vector>
#include <map>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>


namespace learning_topic
{
template <class ContainerAllocator>
struct osd_cmd_
{
  typedef osd_cmd_<ContainerAllocator> Type;

  osd_cmd_()
    : flag()
    , data()  {
    }
  osd_cmd_(const ContainerAllocator& _alloc)
    : flag(_alloc)
    , data(_alloc)  {
  (void)_alloc;
    }



   typedef std::basic_string<char, std::char_traits<char>, typename ContainerAllocator::template rebind<char>::other >  _flag_type;
  _flag_type flag;

   typedef std::vector<uint8_t, typename ContainerAllocator::template rebind<uint8_t>::other >  _data_type;
  _data_type data;





  typedef boost::shared_ptr< ::learning_topic::osd_cmd_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::learning_topic::osd_cmd_<ContainerAllocator> const> ConstPtr;

}; // struct osd_cmd_

typedef ::learning_topic::osd_cmd_<std::allocator<void> > osd_cmd;

typedef boost::shared_ptr< ::learning_topic::osd_cmd > osd_cmdPtr;
typedef boost::shared_ptr< ::learning_topic::osd_cmd const> osd_cmdConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::learning_topic::osd_cmd_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::learning_topic::osd_cmd_<ContainerAllocator> >::stream(s, "", v);
return s;
}


template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator==(const ::learning_topic::osd_cmd_<ContainerAllocator1> & lhs, const ::learning_topic::osd_cmd_<ContainerAllocator2> & rhs)
{
  return lhs.flag == rhs.flag &&
    lhs.data == rhs.data;
}

template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator!=(const ::learning_topic::osd_cmd_<ContainerAllocator1> & lhs, const ::learning_topic::osd_cmd_<ContainerAllocator2> & rhs)
{
  return !(lhs == rhs);
}


} // namespace learning_topic

namespace ros
{
namespace message_traits
{





template <class ContainerAllocator>
struct IsFixedSize< ::learning_topic::osd_cmd_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::learning_topic::osd_cmd_<ContainerAllocator> const>
  : FalseType
  { };

template <class ContainerAllocator>
struct IsMessage< ::learning_topic::osd_cmd_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::learning_topic::osd_cmd_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::learning_topic::osd_cmd_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::learning_topic::osd_cmd_<ContainerAllocator> const>
  : FalseType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::learning_topic::osd_cmd_<ContainerAllocator> >
{
  static const char* value()
  {
    return "cefdc84f6210da72c40569c290984959";
  }

  static const char* value(const ::learning_topic::osd_cmd_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0xcefdc84f6210da72ULL;
  static const uint64_t static_value2 = 0xc40569c290984959ULL;
};

template<class ContainerAllocator>
struct DataType< ::learning_topic::osd_cmd_<ContainerAllocator> >
{
  static const char* value()
  {
    return "learning_topic/osd_cmd";
  }

  static const char* value(const ::learning_topic::osd_cmd_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::learning_topic::osd_cmd_<ContainerAllocator> >
{
  static const char* value()
  {
    return "string flag\n"
"uint8[] data\n"
;
  }

  static const char* value(const ::learning_topic::osd_cmd_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::learning_topic::osd_cmd_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.flag);
      stream.next(m.data);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct osd_cmd_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::learning_topic::osd_cmd_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::learning_topic::osd_cmd_<ContainerAllocator>& v)
  {
    s << indent << "flag: ";
    Printer<std::basic_string<char, std::char_traits<char>, typename ContainerAllocator::template rebind<char>::other > >::stream(s, indent + "  ", v.flag);
    s << indent << "data[]" << std::endl;
    for (size_t i = 0; i < v.data.size(); ++i)
    {
      s << indent << "  data[" << i << "]: ";
      Printer<uint8_t>::stream(s, indent + "  ", v.data[i]);
    }
  }
};

} // namespace message_operations
} // namespace ros

#endif // LEARNING_TOPIC_MESSAGE_OSD_CMD_H