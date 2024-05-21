#ifndef COMMON_REGISTRY_HPP_
#define COMMON_REGISTRY_HPP_

#include "common/prereqs.hpp"
#include "common/predefs.hpp"

namespace sgdlib {

template <typename ObjectPtrType, typename... Args>
class Registry {
public:
    using Creator = std::function<ObjectPtrType(Args...)>;

    Registry() : registry_() { }
    void Register(const std::string& key, Creator creator) {
        std::lock_guard<std::mutex> lock(register_mutex_);
        if (registry_.count(key) != 0) {
            std::ostringstream err_msg;
            err_msg << "key: " << key << " already exists in registry" << std::endl;
            throw std::runtime_error(err_msg.str());
        }
        registry_[key] = creator;
    }
    void Register(const std::string& key, 
                  Creator creator, 
                  const std::string& help_msg) {
        Register(key, creator);
        help_msg_[key] = help_msg;
    }

    ObjectPtrType Create(const std::string& key, Args... args) {
        if (registry_.count(key) == 0) {
            // Returns nullptr if the key is not registered.
            return nullptr;
        }
        return registry_[key](args...);
    }
    std::vector<std::string> Keys() {
        std::vector<std::string> keys;
        for (const auto& it : registry_) keys.push_back(it.first);
        return keys;
    }

    const std::unordered_map<std::string, std::string>& help_msg() const { 
        return help_msg_; 
    }
    const char* help_msg(const std::string& key) const {
        auto it = help_msg_.find(key);
        if (it == help_msg_.end()) {
            return nullptr;
        }
        return it->second.c_str();
    }

protected:
    std::mutex register_mutex_;
    std::unordered_map<std::string, Creator> registry_;
    std::unordered_map<std::string, std::string> help_msg_;

    DISABLE_COPY_AND_ASSIGN(Registry);
};

template <typename  ObjectPtrType, typename ... Args>
class Registerer {
public:
    Registerer(const std::string& key,
                Registry<ObjectPtrType, Args...>* registry,
                typename Registry<ObjectPtrType, Args...>::Creator creator,
                const std::string& help_msg = "") {
        registry->Register(key, creator, help_msg);
    }

    template <typename  DerivedType>
    static ObjectPtrType DefaultCreator(Args... args) {
        return ObjectPtrType(new DerivedType(args...));
    }
};

#define DECLARE_TYPED_REGISTRY(RegistryName, ObjectType, PtrType, ...)            \
    ::sgdlib::Registry<PtrType<ObjectType>, ##__VA_ARGS__>* RegistryName();                 \
    typedef ::sgdlib::Registerer<PtrType<ObjectType>, ##__VA_ARGS__>                        \
       Registerer##RegistryName;

#define DEFINE_TYPED_REGISTRY(RegistryName, ObjectType, PtrType, ...)             \
    ::sgdlib::Registry<PtrType<ObjectType>, ##__VA_ARGS__>* RegistryName() {                \
        static ::sgdlib::Registry<PtrType<ObjectType>, ##__VA_ARGS__>* registry =             \
            new ::sgdlib::Registry<PtrType<ObjectType>, ##__VA_ARGS__>();                       \
        return registry;                                                            \
    }

#define REGISTER_TYPED_CLASS(RegistryName, key, ...)                              \
    namespace {                                                                   \
      static ::sgdlib::Registerer##RegistryName ANONYMOUS_VARIABLE(g_##RegistryName)(       \
        key,                                                                     \
        RegistryName(),                                                          \
        ::sgdlib::Registerer##RegistryName::DefaultCreator<__VA_ARGS__>,                   \
        DemangleType<__VA_ARGS__>());                                            \
    }

#define DECLARE_REGISTRY(RegistryName, ObjectType, ...)                           \
    DECLARE_TYPED_REGISTRY(                                                       \
        RegistryName, ObjectType, std::unique_ptr, ##__VA_ARGS__)

#define DEFINE_REGISTRY(RegistryName, ObjectType, ...)                            \
    DEFINE_TYPED_REGISTRY(                                                        \
        RegistryName, ObjectType, std::unique_ptr, ##__VA_ARGS__)

// key is the name, the second is derived class
#define REGISTER_CLASS(RegistryName, key, ...)                                    \
    REGISTER_TYPED_CLASS(RegistryName, #key, __VA_ARGS__)


} // end namespace

#endif // COMMON_REGISTRY_HPP_