#pragma once
#include "../imports.hpp"
#include "color.hpp"


namespace console {
  struct with_separator {
    explicit with_separator(std::string sep = " ") : sep(std::move(sep)) {}
    std::string sep;
  };

  struct separated_stream {
    separated_stream(std::ostream &stream, std::string sep)
      : Stream(stream), Sep(std::move(sep)) {}

    template<class Rhs>
    fn operator<<(Rhs &&rhs) {
      if (First) First = false;
      else Stream << Sep;

      Stream << std::forward<Rhs>(rhs);
      return *this;
    }

    fn operator<<(std::ostream &(*manip)(std::ostream &)) {
      manip(Stream);
      return *this;
    }

    fn operator<<(const with_separator wsep) {
      Sep = wsep.sep;
      return *this;
    }

  private:
    std::ostream &Stream;
    std::string Sep;
    bool First = true;
  };

  inline fn operator<<(std::ostream &stream, with_separator wsep) {
    return separated_stream(stream, std::move(wsep.sep));
  }


  template<typename ...Args>
  inline fn print(Args &&... args) {
    (std::cout <<  ... <<  std::forward<Args>(args)) << std::endl;
  }

  template<typename ...Args>
  inline fn log(Args &&... args) {
    print(color::Gray, "Log  ", color::Yellow, " : ", color::Gray, args..., with_separator(""),
          color::Yellow, ".",
          color::Reset);
  }

  template<typename ...Args>
  inline fn info(Args &&... args) {
    print(color::Cyan, "Info ", color::Yellow, " : ", color::Cyan, args..., with_separator(""),
          color::Yellow, ".",
          color::Reset);
  }

  template<typename ...Args>
  inline fn event(Args &&... args) {
    print(color::Green, "Event", color::Yellow, " : ", color::Green, args..., with_separator(""),
          color::Yellow, ".",
          color::Reset);
  }

  template<typename ...Args>
  inline fn error(Args &&... args) {
    print(color::Red, "Error", color::Yellow, " : ", color::Red, args..., with_separator(""),
          color::Yellow, ".",
          color::Reset);
  }
}
