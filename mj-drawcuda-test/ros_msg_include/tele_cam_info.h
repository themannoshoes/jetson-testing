// Generated by gencpp from file learning_topic/tele_cam_info.msg
// DO NOT EDIT!


#ifndef LEARNING_TOPIC_MESSAGE_TELE_CAM_INFO_H
#define LEARNING_TOPIC_MESSAGE_TELE_CAM_INFO_H


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
struct tele_cam_info_
{
  typedef tele_cam_info_<ContainerAllocator> Type;

  tele_cam_info_()
    : space(0)
    , photo_captured(0)
    , photo_synced(0)  {
    }
  tele_cam_info_(const ContainerAllocator& _alloc)
    : space(0)
    , photo_captured(0)
    , photo_synced(0)  {
  (void)_alloc;
    }



   typedef uint32_t _space_type;
  _space_type space;

   typedef uint32_t _photo_captured_type;
  _photo_captured_type photo_captured;

   typedef uint32_t _photo_synced_type;
  _photo_synced_type photo_synced;





  typedef boost::shared_ptr< ::learning_topic::tele_cam_info_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::learning_topic::tele_cam_info_<ContainerAllocator> const> ConstPtr;

}; // struct tele_cam_info_

typedef ::learning_topic::tele_cam_info_<std::allocator<void> > tele_cam_info;

typedef boost::shared_ptr< ::learning_topic::tele_cam_info > tele_cam_infoPtr;
typedef boost::shared_ptr< ::learning_topic::tele_cam_info const> tele_cam_infoConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::learning_topic::tele_cam_info_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::learning_topic::tele_cam_info_<ContainerAllocator> >::stream(s, "", v);
return s;
}


template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator==(const ::learning_topic::tele_cam_info_<ContainerAllocator1> & lhs, const ::learning_topic::tele_cam_info_<ContainerAllocator2> & rhs)
{
  return lhs.space == rhs.space &&
    lhs.photo_captured == rhs.photo_captured &&
    lhs.photo_synced == rhs.photo_synced;
}

template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator!=(const ::learning_topic::tele_cam_info_<ContainerAllocator1> & lhs, const ::learning_topic::tele_cam_info_<ContainerAllocator2> & rhs)
{
  return !(lhs == rhs);
}


} // namespace learning_topic

namespace ros
{
namespace message_traits
{





template <class ContainerAllocator>
struct IsFixedSize< ::learning_topic::tele_cam_info_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::learning_topic::tele_cam_info_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::learning_topic::tele_cam_info_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::learning_topic::tele_cam_info_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::learning_topic::tele_cam_info_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::learning_topic::tele_cam_info_<ContainerAllocator> const>
  : FalseType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::learning_topic::tele_cam_info_<ContainerAllocator> >
{
  static const char* value()
  {
    return "2395192a9b0099dc9b6cac552ea3717f";
  }

  static const char* value(const ::learning_topic::tele_cam_info_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0x2395192a9b0099dcULL;
  static const uint64_t static_value2 = 0x9b6cac552ea3717fULL;
};

template<class ContainerAllocator>
struct DataType< ::learning_topic::tele_cam_info_<ContainerAllocator> >
{
  static const char* value()
  {
    return "learning_topic/tele_cam_info";
  }

  static const char* value(const ::learning_topic::tele_cam_info_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::learning_topic::tele_cam_info_<ContainerAllocator> >
{
  static const char* value()
  {
    return "uint32 space            # remaining space\n"
"uint32 photo_captured\n"
"uint32 photo_synced\n"
;
  }

  static const char* value(const ::learning_topic::tele_cam_info_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::learning_topic::tele_cam_info_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.space);
      stream.next(m.photo_captured);
      stream.next(m.photo_synced);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct tele_cam_info_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::learning_topic::tele_cam_info_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::learning_topic::tele_cam_info_<ContainerAllocator>& v)
  {
    s << indent << "space: ";
    Printer<uint32_t>::stream(s, indent + "  ", v.space);
    s << indent << "photo_captured: ";
    Printer<uint32_t>::stream(s, indent + "  ", v.photo_captured);
    s << indent << "photo_synced: ";
    Printer<uint32_t>::stream(s, indent + "  ", v.photo_synced);
  }
};

} // namespace message_operations
} // namespace ros

#endif // LEARNING_TOPIC_MESSAGE_TELE_CAM_INFO_H
